import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import keras
import librosa 
import numpy as np
from functions.feature_extraction import extract_features
import joblib
path = os.getcwd()
# defining paths
model_path = os.path.join(path,'utils/speech_recognition_model.keras')
encoder_path = os.path.join(path, 'utils/encoder.pkl')
scaler_path = os.path.join(path,'utils/scaler.pkl')
# loading model, encoder and scaler
model = keras.saving.load_model(model_path)
encoder = joblib.load(encoder_path)
scaler = joblib.load(scaler_path)

def model_predict():
	fname = os.path.join(path, 'audios/audio_to_process.wav')
	print(fname)
	data_,sample_rate = librosa.load(fname)
	print("done extracting data...",data_)
	X_ = np.array(extract_features(data_))
	print("Extracted Features ✅")
	X_ = scaler.transform(X_.reshape(1, -1))
	print("Scaled Data ✅")
	pred_test_ = model.predict(np.expand_dims(X_, axis=2))
	print("Predicted Data ✅")
	max_pred_index = np.argmax(pred_test_[0])
	max_pred_emotion = encoder.categories_[0][max_pred_index]
	print("Made Predictions ✅")
	print(max_pred_emotion)
	return max_pred_emotion


