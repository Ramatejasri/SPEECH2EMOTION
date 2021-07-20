# SPEECH2EMOTION
# Data Preprocessing
***Run these files Sequntially                               
step0:                                                                                     
pip install -r requirements.txt                                                       
step1:                                                      
cd SPEECH2EMOTION/preprocess                                                                                               
python 1_extract_emotion_labels.py                                     
python 2_build_audio_vectors.py  
python 3_extract_mfcc_feats.py  
python 4_prepare_data.py                                             
python extract_voiced_label.py                                                           
step2:                                                                                   
cd SPEECH2EMOTION/models                                                  
Model Training                                                                    
python lstm_classifier.py                                                                            
step3:                                    
Predictions on test data and performance metrics                                    
python predict.py***

