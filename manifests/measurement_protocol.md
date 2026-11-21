# Measurement protocol (frozen before confirmation)
- Reference: released binary mask (MMOTU `_binary`; BUSI union of `_mask*` files), original image grid, never redrawn or corrected.
- Diameter: maximum Feret = max distance between foreground pixel corners (convex hull of corner points), in pixels; relative error |d_hat - d| / d (unit free, spacing unavailable).
- Prediction: sigmoid -> resize back to original geometry (remove letterbox padding, bilinear) -> threshold 0.5 -> largest connected component (8-conn). All-foreground reported as sensitivity.
- Empty prediction on lesion-positive image: d_hat = 0, relative error 1, counted as missed; failures never excluded.
- Relative error computed on lesion-positive references only; BUSI normal images reported via fraction with any foreground and foreground area fraction.
- Splits: conservative near-duplicate clusters (dHash, Hamming <= 4) as grouping unit; MMOTU official val (469) = benchmark replication test, train images clustering with it removed (19), dev carved from official train; BUSI 70/10/20 by cluster stratified by class (train uses lesion-positive images only).
- Primary endpoint: mean relative error, seeds averaged per image; paired difference vs. strongest comparator frozen on dev; 95% CI by cluster bootstrap (5000).
