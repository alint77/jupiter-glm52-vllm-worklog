**Ship td_v55.** Based on the supplied diffs and reported results, the review-13 conditions are satisfied. I see no remaining kernel blocker within the stated contract.

The map bound closes the placement-less indexing hole. Unsigned epoch advancement, clearing every ready flag, and restarting at 1 address wrap aliasing under the stated PDL ordering. Removing the dead fence leaves the accepted ATOM_AR protocol intact. The expanded adversarial coverage and failing v54 wrap control support sign-off.

Two nonblocking test/reporting notes:

- The wrap test seeds flags with **0**, so it detects the original bug but does not independently establish that clearing is necessary and effective. Also seeding flags with **1** would catch a regression that skips epoch 0 but omits clearing.
- `worst_rel_err` excludes wrap and OOB checks. Include their errors in that accumulator or label the table column “worst regular/graph row error.”

The proposed performance wording is appropriately qualified as **modeled prediction**, and the NOAHEAD increment remains unestablished. Identical layer/finalize SASS supports carrying over their measurements; it does not establish unchanged end-to-end timing for the modified route preparation.
