import json
import time

from pyannote.audio import Pipeline

#Se crea una instancia de la clase Pipeline utilizando el modelo pre-entrenado "pyannote/speaker-diarization" y se asigna el resultado a la variable pipeline. El token se utiliza para autenticarse en el servidor que proporciona el modelo pre-entrenado.
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",
                                    use_auth_token="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")


start_time = time.time()
print("Start time diarization")
print("Start diarization")
diarization = pipeline("mutefire.wav")
listDiarization = []    
speakers = {}
data = {
    "total_segmentos":0 ,
    "total_speakers": 0,
    "total_segmentos_por_speaker": {},
    "segmentos": []
}
print("End diarization")
#se recorre cada track de la diarización utilizando el método itertracks y se asigna cada valor a las variables turn, _ y speaker. El yield_label=True indica que se desea acceder a la etiqueta del hablante en cada iteracion.
print("Start writing json")
for turn, _, speaker in diarization.itertracks(yield_label=True):
    line= {"start": turn.start, "stop": turn.end, "speaker": speaker}    
    listDiarization.append(line)
    if speaker not in speakers:
        speakers[speaker] = 1
    else:
        speakers[speaker] += 1        
#añadir cuantos speakers reconocio y el total de segmentos por speaker
data["total_segmentos"] = len(listDiarization)
data["total_speakers"] = len(speakers)
data["total_segmentos_por_speaker"] = speakers
data["segmentos"] = listDiarization
with open("mutefire.json", "w") as file:
    file.write(json.dumps(data, indent=4))
    
end_time = time.time()
print("End time diarization")
duration = end_time - start_time
#Guardar registro del tiempo de ejecucion en un archivo de texto
print("Duración del proceso: {:.2f} segundos".format(duration))
with open("tiempo_diarizacion.txt", "w") as file:
    file.write("Duración del proceso: {:.2f} segundos".format(duration))
    
print("End writing json" )    