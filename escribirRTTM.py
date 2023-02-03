from pyannote.audio import Pipeline

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",
                                    use_auth_token="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")

# 4. apply pretrained pipeline
diarization = pipeline("tolive.wav")

# 5. print the result

with open("toLive/tolive.rttm", "w") as rttm:
    diarization.write_rttm(rttm)