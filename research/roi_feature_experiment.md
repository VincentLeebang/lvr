# ROI Feature Extraction and Comparison

## Objective
To extract visual features from the complete image and its selected Region of Interest (ROI), and compare their feature representations.

## Methodology
1. Loaded the original image and its annotated bounding box.
2. Extracted the ROI using the verified bounding-box coordinates.
3. Preprocessed both images using resizing and normalization.
4. Used a pretrained ResNet18 model as the feature extractor.
5. Removed the final classification layer to obtain 512-dimensional feature vectors.
6. Compared the feature vectors using cosine similarity.

## Experimental Results

| Parameter | Result |
|---|---|
| Feature extractor | ResNet18 |
| Device | Apple MPS |
| Full image feature shape | (1, 512) |
| ROI feature shape | (1, 512) |
| Cosine similarity | 0.8422 |

## Observation
The full image and ROI produced feature vectors of identical dimensions. The cosine similarity of 0.8422 indicates substantial similarity between their feature representations.

## Conclusion
The experiment successfully extracted and compared visual features from the full image and selected ROI. This provides a preliminary feature-extraction component for further investigation of bounding-box-guided visual reasoning in LATENT-SIGHT.

**Limitation:** This experiment uses ResNet18 independently of the custom Qwen LVR model. It does not yet demonstrate improved visual question answering or integration of ROI features into latent reasoning.