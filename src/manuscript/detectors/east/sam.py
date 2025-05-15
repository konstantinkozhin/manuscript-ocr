import torch


class SAMSolver(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer_cls, rho=0.05, use_adaptive=False, **kwargs):
        if rho < 0:
            raise ValueError("rho must be non-negative")

        self.use_adaptive = use_adaptive  # ✅ ВАЖНО: сохранить как атрибут

        defaults = dict(rho=rho, use_adaptive=use_adaptive, **kwargs)
        super().__init__(params, defaults)

        self._optimizer = base_optimizer_cls(self.param_groups, **kwargs)
        self.param_groups = self._optimizer.param_groups

    @torch.no_grad()
    def step(self, closure):
        assert closure is not None
        closure = torch.enable_grad()(closure)

        # 1-й шаг: считаем loss и градиенты
        loss = closure()
        loss.backward()                     # ✅ чтобы появились p.grad
        self._ascent_step()

        # 2-й шаг: снова loss + градиенты в новой точке
        loss_2 = closure()
        loss_2.backward()                   # ✅ нужно снова
        self._descent_step()

        return loss_2

    @torch.no_grad()
    def _descent_step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # возвращаемся к w

        self._optimizer.step()  # делаем обновление

    @torch.no_grad()
    def _ascent_step(self):
        grad_norm = self._compute_grad_magnitude()
        self._last_grad_norm = grad_norm  # 👈 сохраняем для логгера

        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["use_adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # идём к максимуму

    def _compute_grad_magnitude(self):
        norms = [
            ((torch.abs(p) if self.use_adaptive else 1.0) * p.grad).norm(p=2)
            for group in self.param_groups for p in group["params"]
            if p.grad is not None
        ]
        if len(norms) == 0:
            return torch.tensor(0.0, device=self.param_groups[0]["params"][0].device)
        return torch.norm(torch.stack(norms), p=2)

    def zero_grad(self):
        self._optimizer.zero_grad()

    def state_dict(self):
        return self._optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self._optimizer.load_state_dict(state_dict)
