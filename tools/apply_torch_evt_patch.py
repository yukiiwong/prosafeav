"""Add the EVT safety head and the imagination-time EVT penalty to the PyTorch
ProSafeAV variants, and fix the attribute/method name collision that currently
makes both of them unrunnable.

The collision: ``__init__`` does ``self.policy = LightweightPolicy(...)``, which
shadows the ``def policy(self, obs, state, mode)`` method defined on the same
class.  ``__call__`` then dispatches to ``self.policy(obs, state, mode)``, i.e.
to the *network* with three positional arguments, whose ``forward`` accepts two.
The network attribute is renamed to ``policy_net`` so the method survives.

Idempotent: every replacement checks whether it has already been applied.
"""
import sys

ROOT = "/home/yukai/CarDreamer_prosafeav/dreamerv3"
applied, skipped = [], []


def patch(path, old, new, tag, required=True, already=None):
    """Apply one exact-string replacement.

    ``already`` overrides the idempotency probe.  It is needed where a later
    patch rewrites the same lines, so that ``new`` no longer appears verbatim
    even though the change has in fact been made.
    """
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    if (already or new) in src:
        skipped.append(tag)
        return
    if old not in src:
        if required:
            print(f"ERROR: anchor not found for {tag} in {path}")
            sys.exit(1)
        skipped.append(tag + " (absent)")
        return
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(src.replace(old, new, 1))
    applied.append(tag)


SAFETY_IMPORT = "from evt_torch import EVTImaginationPenalty, SafetyHead\n"

HELPER = '''
    # ------------------------------------------------------------------ #
    # ProSafeAV EVT helpers
    # ------------------------------------------------------------------ #
    def _evt_batch(self, batch, key, dim, default=0.0):
        """Pull an auxiliary observation out of the batch as ``[B, dim]``.

        The environment emits ``safety`` (2-D) and ``evt_params`` (10-D) at every
        step.  Batches may arrive as ``[B, dim]`` or ``[B, T, dim]``; in the
        latter case the first timestep is used, because the EVT parameters are
        constant along a rollout by construction and the safety target is
        aligned with the same step as the observation.
        """
        import numpy as _np

        if key not in batch:
            return torch.full((self._last_batch_size, dim), default, device=self.device)
        value = torch.tensor(_np.asarray(batch[key]), dtype=torch.float32, device=self.device)
        if value.dim() == 3:
            value = value[:, 0]
        return value.reshape(-1, dim)[: self._last_batch_size]

'''


def rename_policy_attribute(path, ctor, policy_call, imag_call, prefix):
    """Rename the shadowing ``self.policy`` network attribute to ``policy_net``.

    ``self.policy_optimizer`` deliberately keeps its name, so the replacements
    are exact strings rather than a blanket substitution.
    """
    for old, new, tag in [
        (ctor, ctor.replace("self.policy =", "self.policy_net ="), f"{prefix}:rename_ctor"),
        ("            self.policy.parameters(),",
         "            self.policy_net.parameters(),", f"{prefix}:rename_opt"),
        (policy_call, policy_call.replace("self.policy(", "self.policy_net("),
         f"{prefix}:rename_policy_call"),
        (imag_call, imag_call.replace("self.policy(", "self.policy_net("),
         f"{prefix}:rename_imag_call"),
        ("            'policy': self.policy.state_dict(),",
         "            'policy': self.policy_net.state_dict(),", f"{prefix}:rename_save"),
        ("        self.policy.load_state_dict(data['policy'])",
         "        self.policy_net.load_state_dict(data['policy'])", f"{prefix}:rename_load"),
    ]:
        patch(path, old, new, tag)


def add_save_load_safety(path, prefix):
    patch(
        path,
        "            'policy': self.policy_net.state_dict(),",
        "            'policy': self.policy_net.state_dict(),\n"
        "            'safety_head': self.safety_head.state_dict(),",
        f"{prefix}:save_safety",
    )
    patch(
        path,
        "        self.policy_net.load_state_dict(data['policy'])",
        "        self.policy_net.load_state_dict(data['policy'])\n"
        "        if 'safety_head' in data:\n"
        "            self.safety_head.load_state_dict(data['safety_head'])",
        f"{prefix}:load_safety",
    )


def add_optimizer_and_clip(path, prefix, lr_default):
    patch(
        path,
        f"            list(self.reward_model.parameters()),\n"
        f"            lr=config.get('model_lr', {lr_default})",
        f"            list(self.reward_model.parameters()) +\n"
        f"            list(self.safety_head.parameters()),\n"
        f"            lr=config.get('model_lr', {lr_default})",
        f"{prefix}:safety_in_optimizer",
        already="list(self.safety_head.parameters())",
    )
    patch(
        path,
        "            list(self.reward_model.parameters()),\n            10.0\n        )",
        "            list(self.reward_model.parameters()) +\n"
        "            list(self.safety_head.parameters()),\n            10.0\n        )",
        f"{prefix}:safety_in_clip",
    )


# --------------------------------------------------------------------------- #
def patch_rssm():
    path = f"{ROOT}/prosafeav_rssm_agent.py"
    latent = "torch.cat([h_imag, z_imag], dim=-1)"

    patch(path, "class LightweightEncoder(nn.Module):",
          SAFETY_IMPORT + "\n\nclass LightweightEncoder(nn.Module):", "rssm:import")

    rename_policy_attribute(
        path,
        ctor="self.policy = LightweightPolicy(",
        policy_call="            logits = self.policy(h, z)",
        imag_call="            action_logits = self.policy(h_imag, z_imag)",
        prefix="rssm",
    )

    patch(
        path,
        "        self.policy_net = LightweightPolicy(self.stochastic_dim + self.deterministic_dim, self.act_dim).to(self.device)",
        "        self.policy_net = LightweightPolicy(self.stochastic_dim + self.deterministic_dim, self.act_dim).to(self.device)\n"
        "        # ProSafeAV: predicts the normalised (TTC, DRAC) pair from the latent\n"
        "        # state so the EVT tail risk can be evaluated on imagined rollouts.\n"
        "        self.safety_head = SafetyHead(self.stochastic_dim + self.deterministic_dim, hidden=64).to(self.device)\n"
        "        self.evt = EVTImaginationPenalty(config)\n"
        "        self._last_batch_size = 1",
        "rssm:safety_head",
    )
    add_optimizer_and_clip(path, "rssm", "1e-3")

    patch(
        path,
        "        pred_reward = self.reward_model(h, z).squeeze(-1)\n"
        "        reward_loss = F.mse_loss(pred_reward, rewards)",
        "        pred_reward = self.reward_model(h, z).squeeze(-1)\n"
        "        reward_loss = F.mse_loss(pred_reward, rewards)\n\n"
        "        # ProSafeAV: regress the surrogate safety measures from the latent state.\n"
        "        self._last_batch_size = h.size(0)\n"
        "        safety_target = self._evt_batch(batch, 'safety', 2)\n"
        "        pred_safety = self.safety_head(torch.cat([h, z], dim=-1))\n"
        "        safety_loss = F.mse_loss(pred_safety, safety_target)",
        "rssm:safety_loss",
    )
    patch(
        path,
        "        model_loss = recon_loss + self.kl_weight * kl_loss + reward_loss",
        "        model_loss = recon_loss + self.kl_weight * kl_loss + reward_loss + safety_loss",
        "rssm:model_loss",
    )
    patch(
        path,
        "        imagined_reward = 0\n        h_imag, z_imag = h_start.detach(), z_start.detach()",
        "        imagined_reward = 0\n        imagined_risk = 0\n"
        "        evt_params = self._evt_batch(batch, 'evt_params', 10)\n"
        "        h_imag, z_imag = h_start.detach(), z_start.detach()",
        "rssm:imag_init",
    )
    patch(
        path,
        "            reward = self.reward_model(h_imag, z_imag).squeeze(-1)\n"
        "            imagined_reward += reward",
        "            reward = self.reward_model(h_imag, z_imag).squeeze(-1)\n\n"
        "            # ProSafeAV: subtract the EVT tail risk of the *imagined* state.\n"
        f"            risk = self.evt.risk(self.safety_head({latent}), evt_params)\n"
        "            imagined_risk = imagined_risk + risk\n"
        "            imagined_reward += reward - self.evt.weight * risk",
        "rssm:imag_penalty",
    )
    patch(
        path,
        "            'reward_loss': reward_loss.item(),\n            'policy_loss': policy_loss.item(),",
        "            'reward_loss': reward_loss.item(),\n"
        "            'safety_loss': safety_loss.item(),\n"
        "            'imag_evt_risk': float(imagined_risk.mean()) if torch.is_tensor(imagined_risk) else 0.0,\n"
        "            'policy_loss': policy_loss.item(),",
        "rssm:metrics",
    )
    add_save_load_safety(path, "rssm")
    patch(path, "    def _preprocess_obs(self, obs):",
          HELPER + "    def _preprocess_obs(self, obs):", "rssm:helper")


# --------------------------------------------------------------------------- #
def patch_deterministic():
    path = f"{ROOT}/prosafeav_deterministic_agent.py"

    patch(path, "class DeterministicEncoder(nn.Module):",
          SAFETY_IMPORT + "\n\nclass DeterministicEncoder(nn.Module):", "det:import")

    rename_policy_attribute(
        path,
        ctor="self.policy = DeterministicPolicy(",
        policy_call="            logits = self.policy(h)",
        imag_call="            action_logits = self.policy(h_imag)",
        prefix="det",
    )

    patch(
        path,
        "        self.policy_net = DeterministicPolicy(self.hidden_dim, self.act_dim).to(self.device)",
        "        self.policy_net = DeterministicPolicy(self.hidden_dim, self.act_dim).to(self.device)\n"
        "        # ProSafeAV: safety head over the deterministic latent.\n"
        "        self.safety_head = SafetyHead(self.hidden_dim, hidden=64).to(self.device)\n"
        "        self.evt = EVTImaginationPenalty(config)\n"
        "        self._last_batch_size = 1",
        "det:safety_head",
    )
    add_optimizer_and_clip(path, "det", "1e-3")

    # The latent-consistency term compares the GRU hidden state against the
    # observation embedding, but hidden_dim (128) and embed_dim (64) differ by
    # default, so the loss raised a shape error for every configuration -- this
    # agent could never have completed a training step.  A learned projection
    # into the embedding space fixes it without changing the architecture.
    patch(
        path,
        "        self.safety_head = SafetyHead(self.hidden_dim, hidden=64).to(self.device)",
        "        self.safety_head = SafetyHead(self.hidden_dim, hidden=64).to(self.device)\n"
        "        self.consistency_proj = nn.Linear(self.hidden_dim, self.embed_dim).to(self.device)",
        "det:consistency_proj",
    )
    patch(
        path,
        "            list(self.safety_head.parameters()),\n"
        "            lr=config.get('model_lr', 1e-3)",
        "            list(self.safety_head.parameters()) +\n"
        "            list(self.consistency_proj.parameters()),\n"
        "            lr=config.get('model_lr', 1e-3)",
        "det:proj_in_optimizer",
    )
    patch(
        path,
        "        consistency_loss = F.mse_loss(h_next, next_embed.detach())",
        "        consistency_loss = F.mse_loss(self.consistency_proj(h_next), next_embed.detach())",
        "det:consistency_fix",
    )

    patch(
        path,
        "        pred_reward = self.reward_model(h_next).squeeze(-1)\n"
        "        reward_loss = F.mse_loss(pred_reward, rewards)",
        "        pred_reward = self.reward_model(h_next).squeeze(-1)\n"
        "        reward_loss = F.mse_loss(pred_reward, rewards)\n\n"
        "        # ProSafeAV: regress the surrogate safety measures from the latent state.\n"
        "        self._last_batch_size = h_next.size(0)\n"
        "        safety_target = self._evt_batch(batch, 'safety', 2)\n"
        "        pred_safety = self.safety_head(h_next)\n"
        "        safety_loss = F.mse_loss(pred_safety, safety_target)",
        "det:safety_loss",
    )
    patch(
        path,
        "        model_loss = recon_loss + 0.1 * consistency_loss + reward_loss",
        "        model_loss = recon_loss + 0.1 * consistency_loss + reward_loss + safety_loss",
        "det:model_loss",
    )
    patch(
        path,
        "        imagined_reward = 0\n        h_imag = h_start.detach()",
        "        imagined_reward = 0\n        imagined_risk = 0\n"
        "        evt_params = self._evt_batch(batch, 'evt_params', 10)\n"
        "        h_imag = h_start.detach()",
        "det:imag_init",
    )
    patch(
        path,
        "            reward = self.reward_model(h_imag).squeeze(-1)\n"
        "            imagined_reward += reward",
        "            reward = self.reward_model(h_imag).squeeze(-1)\n\n"
        "            # ProSafeAV: subtract the EVT tail risk of the *imagined* state.\n"
        "            risk = self.evt.risk(self.safety_head(h_imag), evt_params)\n"
        "            imagined_risk = imagined_risk + risk\n"
        "            imagined_reward += reward - self.evt.weight * risk",
        "det:imag_penalty",
    )
    patch(
        path,
        "            'reward_loss': reward_loss.item(),\n            'policy_loss': policy_loss.item(),",
        "            'reward_loss': reward_loss.item(),\n"
        "            'safety_loss': safety_loss.item(),\n"
        "            'imag_evt_risk': float(imagined_risk.mean()) if torch.is_tensor(imagined_risk) else 0.0,\n"
        "            'policy_loss': policy_loss.item(),",
        "det:metrics",
    )
    add_save_load_safety(path, "det")
    patch(path, "    def _preprocess_obs(self, obs):",
          HELPER + "    def _preprocess_obs(self, obs):", "det:helper")


if __name__ == "__main__":
    patch_rssm()
    patch_deterministic()
    print("applied:", applied)
    print("already present:", skipped)
