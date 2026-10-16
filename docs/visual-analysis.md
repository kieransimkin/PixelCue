# Visual analysis contract

PixelCue owns the engine; PrismMoves owns visual-language evidence organisation and reporting. Use the service or the public `pixelcue.visual_analysis.analyse_image` / `pixelcue.visual_service.VisualAnalysisService.analyse` APIs rather than copying numerical or model code.

## Measurement conditions

Input is one explicit local image, at most 128 MiB and 40 million decoded pixels. PixelCue reads one bounded byte snapshot, hashes it, uses frame zero, honours EXIF orientation, converts a usable embedded ICC profile to sRGB, and discloses failed conversion. Without an ICC profile it assumes sRGB. Alpha is composited on an explicit matte (default white); hidden RGB is not treated as visible evidence. Analysis uses a bounded raster (768 longest edge by default, maximum 2048); tiny images are not enlarged. The receipt records source and analysis sizes, matte, alpha fractions sampled at analysis resolution, colour handling and a digest of the analysed raster.

Colour uses deterministic farthest-initialised Lab k-means, at most 8192 samples, six centres and 20 iterations. Reported colours are approximate mean sRGB samples, with cluster fractions. Hue bins are weighted by saturation; achromatic samples supply no hue evidence. Relative luminance linearises sRGB; quantiles, standard deviation and stated dark/light thresholds are measurements. Palette-pair contrast ratios do not prove text/background adjacency, legibility or WCAG compliance.

Texture combines grayscale Shannon entropy, an 8-neighbour local binary-pattern histogram and a symmetric 16-level co-occurrence matrix pooled over four one-pixel directions. Co-occurrence contrast, homogeneity and energy are descriptors, not material labels. Finite-difference gradients provide edge density (threshold 0.08 in gamma-encoded grayscale) and eight orientation bins. Mirror errors measure grayscale pixel differences, not semantic symmetry. A spatial grid retains luminance, saturation, edge density and signal-novelty mass. Mean-removed FFT power is divided into stated cycles-per-pixel bands; resizing and boundary effects influence it. Spectral residual yields a novelty centroid; it is not human attention, subject segmentation or guaranteed safe space. Uniform rasters have no novelty centroid.

## Model receipts

Models are lazy and selected explicitly; statistics require only NumPy and Pillow. Default pretrained profiles are CLIP ViT-B/32, DINOv2-small and BLIP base; a local checkpoint directory can override a selected profile. Models run sequentially and are released between adapters. Local-only loading is the default and `trust_remote_code=False` is enforced. Model receipts include selected checkpoint, resolved revision when the framework supplies it, dependency versions, device, elapsed time, interpretation class, status and failure reason. Local custom checkpoints may have no resolved hub revision; comparisons then deliberately omit embedding similarity.

- **CLIP:** image/text cosine similarities within separate medium, light, material, tone and setting prompt groups; full prompt bank retained. A top prompt is a candidate hypothesis, not a recognised fact. Similarity is not calibrated confidence or probability. The image encoder applies its processor's normal resizing/cropping after the analysis raster.
- **DINOv2:** normalised CLS embedding only; no invented labels. Compare only matching model/checkpoint/revision/preprocessing.
- **BLIP:** deterministic bounded caption generation; captions can hallucinate text, subjects, intent and materials.
- **OCR:** optional pytesseract plus external Tesseract; text, engine-specific confidence and boxes in analysis-raster coordinates; 30-second OCR timeout. No font identification.
- **Existing VLMs:** `VisualAnalysisService` delegates tags to the existing `KeywordExtractionService`. Its original model cache/download behaviour is retained. Consequently `keyword_models` require explicit `allow_download=True`, even when a checkpoint might already be cached. This avoids silently expanding a model-free analysis into a download.

Hosted inference is not used. No face identity or demographic analysis is invoked. Model licences, privacy and hardware suitability require separate review. Optional failed adapters remain unavailable receipts; they are never turned into negative evidence. Requested adapter failures do not discard completed statistics.

Optional real-interface tests use tiny randomly initialised local CLIP, DINOv2 and BLIP checkpoints. Set `PIXELCUE_TEST_MODEL_INTERFACES=1` and run `pytest tests/test_visual_models_runtime.py` in an existing compatible Torch/Transformers runtime. They do not download weights or evaluate pretrained semantic accuracy. The normal model-free suite skips this module.

## Comparison

`compare_profiles` returns pairwise saturation differences and hue/LBP Jensen–Shannon divergence in bits, plus same-file and same-analysis-raster flags. An achromatic histogram gives `null` hue divergence. Model cosine similarity requires matching named checkpoint, a known matching revision and equal vector dimensions. Same raster is only a lineage candidate. Crops, artist approval, intentional variation and independent authorship need reviewed evidence. Input matte/resize/model settings should be matched before drawing conclusions from distances; no combined identity or aesthetic score is supplied.

## Potential problems

- Older PixelCue services have no `/v1/analysis`; inspect `/health` before calling the route.
- Missing Torch/Transformers weights or Tesseract produce per-model unavailable receipts; review them instead of inferring a semantic result.
- A model load may exceed an HTTP client's timeout; timeout does not cancel the server's background thread.
- Measurements at a different matte or resolution can shift. Re-run controlled comparisons with matching options.
- ICC conversion failure, tiny rasters and compressed exports limit precision. Keep the original and inspect actual artwork.
- This analysis route does not use PixelCue's persistent scan cache. Each explicit request is recomputed; keyword tagger reuse stays within the existing keyword service.

Adapter references: [CLIP](https://huggingface.co/docs/transformers/model_doc/clip), [DINOv2](https://huggingface.co/docs/transformers/model_doc/dinov2), [BLIP](https://huggingface.co/docs/transformers/model_doc/blip). These interfaces do not establish accuracy on campaign artwork.
