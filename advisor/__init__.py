"""PufferRoyale advisor: read a live Clash Royale screen, rebuild the policy's observation, and show the
policy's suggested plays to a human player (who taps them, or not). Nothing here sends input to the game.

Layers (each one testable on its own; docs/ADVISOR.md has the plan and the verification record):
  engine_db  -- facts about units / projectiles / cards read from the compiled engine's card DB
  board      -- BoardState: everything the observation needs, in the viewer's own frame
  encoder    -- BoardState -> the exact SPEC §9 observation vector (checked against Game.obs)
"""
