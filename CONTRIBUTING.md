# Contributing to MACRO

MACRO preserves the simulation methods accompanying the cited ICC 2021 paper. Keep proposed changes focused, document their scientific motivation, and avoid altering the historical citation or license information.

For additions, describe the affected model, assumptions, configuration changes, and intended output in the pull request.

For verification, run `python -m unittest discover -s tests -v`,
`python run.py --quick --validate`, and `python run.py --quick`. Keep the agent count compatible
with the concentric layer geometry and choose a frame count divisible by the
number of jumps. Check finite trajectories/attitudes, recorded stages, neighbor
coverage, and the exported HTML/PNG files. Compare numerical outputs before and
after refactors; avoid changing scientific defaults as part of cleanup.

Also run `python run.py --no-plots` when changing numerical integration or attitude behavior. The supplied scenarios explicitly use `attitude.mode: tracking` because the historical norm-error law overflows on long runs. Keep `legacy` available for comparison and document controller/model changes as such. Do not label finite integration or successful export as achieved consensus, collision avoidance, or physical validation.

Output samples are interval endpoints (`dt` through `num_frames * dt`); phase initial states are exported separately. Preserve spacecraft identities, unique target assignments, actual position/velocity handoffs, and the inclusive-start/exclusive-end stage boundaries. A scenario with no exchanges still has a complete holding/attitude interval. New runs must preserve previous run directories and report failure/interruption honestly.
