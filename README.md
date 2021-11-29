# Blended Diffusion for Text-driven Editing of Natural Images

**Blended Diffusion for Text-driven Editing of Natural Images**<br>
Omri Avrahami, Dani Lischinski, Ohad Fried<br>

<!-- Paper: http://todo<br>
Project Page: https://todo<br> -->

Abstract: *Natural language offers a highly intuitive interface for image editing. In this paper, we introduce the first solution for performing local (region-based) edits in generic natural images, based on a natural language description along with an ROI mask.
We achieve our goal by leveraging and combining a pretrained language-image model (CLIP), to steer the edit towards a user-provided text prompt, with a denoising diffusion probabilistic model (DDPM) to generate natural-looking results.
To seamlessly fuse the edited region with the unchanged parts of the image, we spatially blend noised versions of the input image with the local text-guided diffusion latent at a progression of noise levels.
In addition, we show that adding augmentations to the diffusion process mitigates adversarial results.
We compare against several baselines and related methods, both qualitatively and quantitatively, and show that our method outperforms these solutions in terms of overall realism, ability to preserve the background and matching the text. Finally, we show several text-driven editing applications, including adding a new object to an image, removing/replacing/altering existing objects, background replacement, and image extrapolation.*

## Applications

### Multiple synthesis results for the same prompt
<img src="assets/multiple_predictions.jpg" width="800">

### Synthesis results for different prompts
<img src="assets/different_prompts1.jpg" width="800">
<img src="assets/different_prompts2.jpg" width="800">

### Altering part of an existing object
<img src="assets/altering_existing.jpg" width="800">

### Background replacement
<img src="assets/background_replacement1.jpg" width="800">
<img src="assets/background_replacement2.jpg" width="800">

### Scribble-guided editing
<img src="assets/scribble_guided_editing.jpg" width="800">

### Text-guided extrapolation
<img src="assets/extrapolation.jpg" width="800">

### Composing several applications
<img src="assets/composition1.jpg" width="800">
<img src="assets/composition2.jpg" width="800">

## Code availability
Full code will be released soon.