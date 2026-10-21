"""
Put one file per detector in this folder. Every module here is imported when
the API starts, so registering is just:

    from detect import video_detector, frame_detector

    @video_detector("c2_loitering")
    def run(video_path, source_id):
        return [...]   # list of IncidentEvent

Files starting with "_" are skipped — `_example_c4.py` shows the full shape
without being loaded. Copy it, rename it, and fill in your model.
"""
