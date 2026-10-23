"""Read-only Component 03 consumer for the upstream Component 02 tracker.

Does not import C2 files or run another ByteTrack. Component 02 controls IDs;
Component 03 owns only risk-model state and optional pose *estimation*.
Keep a distinct adapter instance for each camera; do not share RNN states.
"""
class C3UpstreamAdapter:
    def __init__(self,weights,camera_id,pose_weights=None):
        from src.live import C3Live  # load torch only when instantiating the real model
        self.engine=C3Live(weights,camera_id=camera_id,pose_weights=None)
        self.camera_id=camera_id
        self.pose_enricher=None
        if pose_weights:
            from src.pose_enrichment import PoseEnricher
            self.pose_enricher=PoseEnricher(pose_weights)

    def push(self,timestamp_s,frame,kps,boxes,track_ids):
        """Use when upstream already provides COCO-17 poses + C2 IDs."""
        self.engine.append_tracked(timestamp_s,frame,kps,boxes,track_ids)

    def push_from_c2(self,timestamp_s,frame,boxes,track_ids):
        """Use when C2 produces only tracked boxes and IDs, but not poses."""
        if self.pose_enricher is None:
            raise RuntimeError('C2 supplies only boxes; configure pose_weights for one-time pose enrichment')
        kps=self.pose_enricher.estimate(frame,boxes)
        self.push(timestamp_s,frame,kps,boxes,track_ids)

    def assess(self,timestamp_s,width,height,audio_wave=None):
        return self.engine.assess(timestamp_s,(width,height),audio_wave)
