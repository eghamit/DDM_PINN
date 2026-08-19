"""Training loop for a single operating point.

:class:`Trainer` optimises the network weights so the composite loss (PDE
residuals + ohmic boundary targets) is minimised at one fixed bias.  It runs

1. an **Adam** phase - robust global descent, with the collocation points
   resampled every ``resample_every`` steps so the residual is enforced across
   the whole domain, then
2. an optional **L-BFGS** phase - a quasi-Newton polish on a fixed point set
   that drives the residual down the last few orders of magnitude (L-BFGS is
   very effective for the smooth, low-dimensional PINN loss once Adam is close).

A bias point is reached by warm-starting from the previous solution (the solver
handles continuation), so each :meth:`solve_point` call is a short refinement,
not a cold start.
"""

import numpy as np
import torch

from DDM_PINN.training.losses import total_loss


class Trainer:
    def __init__(self, physics, sampler, doping, contacts, dtype=torch.float64,
                 num_collocation=4000, adam_steps=800, adam_lr=1e-3,
                 lbfgs_steps=800, resample_every=0, grad_clip=None,
                 verbose=False, log_every=400):
        self.physics = physics
        self.sampler = sampler
        self.doping = doping
        self.contacts = contacts
        self.dtype = dtype
        self.num_collocation = num_collocation
        self.adam_steps = adam_steps
        self.adam_lr = adam_lr
        self.lbfgs_steps = lbfgs_steps
        self.resample_every = resample_every
        self.grad_clip = grad_clip
        self.verbose = verbose
        self.log_every = log_every

    # -- helpers -----------------------------------------------------------
    def _C_at(self, X):
        C = self.doping.scaled(X.detach().cpu().numpy().reshape(-1))
        return torch.tensor(C.reshape(-1, 1), dtype=self.dtype)

    def _boundary(self, scaled_voltages):
        Xb = self.sampler.boundary_points()
        rows = []
        # align each boundary coordinate with the contact sitting there
        pos_to_contact = {round(c.position, 9): c for c in self.contacts.values()}
        for x in Xb.reshape(-1).tolist():
            contact = pos_to_contact[round(x, 9)]
            V = scaled_voltages.get(contact.name, 0.0)
            rows.append(contact.targets(V))
        targets = torch.tensor(np.array(rows), dtype=self.dtype)
        return Xb.to(self.dtype), targets

    # -- main solve --------------------------------------------------------
    def solve_point(self, net, scaled_voltages, weights):
        """Refine ``net`` to the given scaled terminal voltages; return history."""
        Xb, targets = self._boundary(scaled_voltages)
        X = self.sampler.interior(self.num_collocation)
        C = self._C_at(X)
        history = []

        opt = torch.optim.Adam(net.parameters(), lr=self.adam_lr)
        # geometric LR decay to ~1/20 of the initial rate: large early steps to
        # find the depletion shape, small late steps to settle the stiff Poisson
        # residual in the neutral bulk without oscillating.
        sched = torch.optim.lr_scheduler.ExponentialLR(
            opt, gamma=0.05 ** (1.0 / max(self.adam_steps, 1)))
        for step in range(1, self.adam_steps + 1):
            if self.resample_every and step % self.resample_every == 0:
                X = self.sampler.interior(self.num_collocation)
                C = self._C_at(X)
            opt.zero_grad()
            loss, parts = total_loss(self.physics, net, X, C, Xb, targets,
                                     weights)
            loss.backward()
            # optional global-norm gradient clip (off by default: Adam is
            # essentially invariant to a uniform gradient rescale, so clipping
            # mainly matters if a different optimiser is swapped in)
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(net.parameters(), self.grad_clip)
            opt.step()
            sched.step()
            if self.verbose and (step == 1 or step % self.log_every == 0):
                self._log(step, loss, parts)
            history.append(float(loss.detach()))

        if self.lbfgs_steps:
            # L-BFGS is the workhorse for this smooth 1-D BVP: on a FIXED
            # collocation grid the objective is deterministic and the
            # quasi-Newton curvature information drives the stiff Poisson
            # residual down far past what (scale-invariant, noisy) Adam reaches.
            # Reuse the Adam grid when not resampling so the objective is stable.
            if self.resample_every:
                X = self.sampler.interior(self.num_collocation)
                C = self._C_at(X)
            n_rounds = 5
            per = max(1, self.lbfgs_steps // n_rounds)
            for r in range(n_rounds):
                opt = torch.optim.LBFGS(
                    net.parameters(), max_iter=per,
                    line_search_fn="strong_wolfe", tolerance_grad=1e-14,
                    tolerance_change=1e-16, history_size=50,
                )

                def closure():
                    opt.zero_grad()
                    loss, _ = total_loss(self.physics, net, X, C, Xb, targets,
                                         weights)
                    loss.backward()
                    return loss

                opt.step(closure)
                final, parts = total_loss(self.physics, net, X, C, Xb, targets,
                                          weights)
                if self.verbose:
                    self._log(f"LBFGS {(r + 1) * per}", final, parts)
                history.append(float(final.detach()))

        return history

    def _log(self, step, loss, parts):
        msg = "  ".join(f"{k}={v.item():.2e}" for k, v in parts.items())
        print(f"    [{step}] loss={float(loss.detach()):.3e}  {msg}")
