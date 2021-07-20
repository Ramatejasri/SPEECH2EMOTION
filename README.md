# SPEECH2EMOTION
# Data Preprocessing
***Run these files Sequntially                               
step0:                                                                                     
pip install -r requirements.txt                                                       
step1:                                                      
cd SPEECH2EMOTION/preprocess                                                                                               
preprocess/1_extract_emotion_labels.py                                     
      preprocess/2_build_audio_vectors.py  
      preprocess/3_extract_mfcc_feats.py  
      preprocess/4_prepare_data.py                                             
      preprocess/extract_voiced_label.py                                                           
      step2:                                                                                   
      cd SPEECH2EMOTION/models                                                  
      Model Training                                                                    
      Run models/lstm_classifier.py                                                                            
      step3:                                    
      Predictions on test data and performance metrics                                    
      Run models/predict.py***

