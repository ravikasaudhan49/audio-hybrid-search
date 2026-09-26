"""Provider normalization and speaker naming, from tiny synthetic responses (no API calls)."""
from audiosearch.segment import speaker_names
from audiosearch.transcribe import normalize_assemblyai, normalize_deepgram


def test_assemblyai_with_speaker_identification_keeps_names_and_seconds():
    resp = {"audio_duration": 3, "speech_understanding": {"response": {"speaker_identification": {
        "status": "success", "mapping": {"A": "Gary Jordan", "B": "Patrick O'Neill"}}}},
        "words": [{"text": "Hi,", "start": 500, "end": 900, "speaker": "Gary Jordan", "confidence": 0.9},
                  {"text": "Patrick.", "start": 900, "end": 1400, "speaker": "Gary Jordan", "confidence": 0.9},
                  {"text": "Hello.", "start": 1600, "end": 2000, "speaker": "Patrick O'Neill", "confidence": 0.9}]}
    t = normalize_assemblyai(resp)
    assert t["speakers"] == {"0": "Gary Jordan", "1": "Patrick O'Neill"}
    assert t["words"][0] == {"text": "Hi,", "start": 0.5, "end": 0.9, "speaker": 0, "confidence": 0.9}
    assert t["words"][2]["speaker"] == 1


def test_assemblyai_without_identification_has_no_names():
    resp = {"audio_duration": 1, "words": [{"text": "Hi.", "start": 0, "end": 500, "speaker": "A"}]}
    assert normalize_assemblyai(resp)["speakers"] == {"0": None}


def test_deepgram_numbers_speakers_by_first_appearance_without_names():
    resp = {"metadata": {"duration": 2}, "results": {"channels": [{"alternatives": [{"words": [
        {"word": "hi", "punctuated_word": "Hi.", "start": 0, "end": 0.4, "speaker": 1},
        {"word": "yes", "punctuated_word": "Yes.", "start": 0.5, "end": 0.9, "speaker": 0}]}]}]}}
    t = normalize_deepgram(resp)
    assert [w["speaker"] for w in t["words"]] == [0, 1] and t["speakers"] == {"0": None, "1": None}
    assert t["words"][0]["text"] == "Hi."


def test_speaker_names_follow_cleaned_labels_when_a_stray_third_speaker_is_folded():
    t = {"speakers": {"0": "Host", "1": "Guest", "2": "Launch audio"}, "words": [
        {"text": w, "start": i, "end": i + 0.5, "speaker": s}
        for i, (w, s) in enumerate([("Hello", 0), ("there.", 0), ("T-0.", 2), ("Thanks", 1), ("for", 1),
                                    ("having", 1), ("me.", 1), ("Sure.", 0)])]}
    assert speaker_names(t) == {0: "Host", 1: "Guest"}
