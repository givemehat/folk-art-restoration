# Human Evaluation Rubric for Folk Art Restoration

Qualitative assessment is critical because metrics like PSNR and SSIM do not always capture perceptual style consistency. Provide this rubric to evaluators (e.g., via a Google Form or MTurk).

## Evaluation Criteria

Rate each restored image on a scale of 1 to 5 for the following categories:

### 1. Structural Integrity
*How well did the model reconstruct the missing geometry and structural lines?*
- **1:** Completely destroyed structure; hallucinated wrong objects.
- **3:** Acceptable structure, but some lines are misaligned or blurry.
- **5:** Perfect structural reconstruction; impossible to tell it was damaged.

### 2. Style Consistency (Folk Art Specific)
*Does the inpainted region match the specific folk art style (e.g., Madhubani patterns, Warli stick figures)?*
- **1:** Style is completely ignored; looks like generic blur or modern painting.
- **3:** Style is somewhat matched, but brush strokes/patterns look slightly unnatural.
- **5:** Seamless style matching; identical to the original artist's technique.

### 3. Color and Texture Blending
*How well does the restored region blend with the surrounding colors and texture?*
- **1:** Severe color mismatch; obvious borders around the restored region.
- **3:** Colors are close, but slight lighting/texture differences are noticeable.
- **5:** Perfect blending; lighting and texture are uniform across the image.

## Survey Template
**Image Pair ID:** `[Insert ID]`
- Structural Integrity: `[ 1 / 2 / 3 / 4 / 5 ]`
- Style Consistency: `[ 1 / 2 / 3 / 4 / 5 ]`
- Color/Texture: `[ 1 / 2 / 3 / 4 / 5 ]`
- Additional Comments: ___________________________
