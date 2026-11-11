CamVid (Brostow et al., PRL 2009), all 701 labelled frames, 960x720 -> 120x90 (bilinear image, nearest label); the 32 classes grouped into the 11 of the SegNet benchmark (prepare_data.py CAMVID11), Void -> -100 (ignored). Archive: https://s3.amazonaws.com/fast-ai-imagelocal/camvid.tgz.

Arrays: X (701, 90, 120, 3) uint8, y (701, 90, 120) int64.
