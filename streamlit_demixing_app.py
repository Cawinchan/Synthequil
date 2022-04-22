import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import DemixingAudioDataset
from unet_model import UNet
from torch.utils.data import DataLoader, random_split
from utils import save_model, load_model, negative_SDR
import time
import math
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

INSTRUMENTS = ("bass", "drums", "vocals", "other")
import streamlit as st

st.title("50.039: Theory and Practice of Deep Learning - Audio Demixing Project")
st.subheader("Input audio (.wav)")
user_input = {}
user_input["input_audio"] = st.file_uploader(
    "Pick an audio to test"
)

if user_input["input_audio"]:
    st.write("Original audio input")
    st.audio(user_input["input_audio"])

    

## Evaluation Metrics 


## Validation Process


# def main(dataset_dir, test, custom_test_dir, train_checkpoint_dir, model, epoch_count):
    
#     # Get input directory, checkpoint directory, test model path
#     input_dir = None
#     if not test:
#         if dataset_dir==None:
#             raise Exception("Error: no dataset specified for training, please use --dataset-dir for this")
#         input_dir = os.path.join(dataset_dir,"train")
#         if not os.path.isdir(train_checkpoint_dir):
#             os.mkdir(train_checkpoint_dir)
#     else:
#         if dataset_dir==None and custom_test_dir==None:
#             raise Exception("Error: no directory specified for testing, please use either --dataset-dir or --custom-test-dir")
#         input_dir = custom_test_dir if custom_test_dir!=None else os.path.join(dataset_dir,"test")
#         if model==None:
#             raise Exception("Error: no test model specified, please use --model for this")

#     # Toggle train/test mode
#     is_train = not test

#     # Get Dataset object
#     audio_dataset = DemixingAudioDataset(input_dir)
#     train_len = int(0.8*len(audio_dataset))
#     train_dataset, test_dataset = random_split(audio_dataset,[train_len,len(audio_dataset)-train_len],
#     generator=torch.Generator().manual_seed(100)) if is_train else (None, audio_dataset)

#     # Get DataLoader objects
#     train_dataloader = DataLoader(train_dataset,shuffle=True)
#     test_dataloader = DataLoader(test_dataset,shuffle=True)

#     # Get device to load samples and model to
#     device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#     # Define model and optimizer
#     audio_model = nn.DataParallel(UNet([2**i for i in range(1,3)],5,"leaky_relu",INSTRUMENTS))
#     # print(sum(p.numel() for p in audio_model.parameters() if p.requires_grad))
#     # optimizer = optim.SGD(audio_model.parameters(),lr=0.1,momentum=0.9)
#     optimizer = optim.Adam(audio_model.parameters(),lr=0.001) # optimiser recommended by wave-u-net paper

#     # Define loss criterion
#     criterion = negative_SDR()


#     with torch.no_grad():
#                 for i in test_dataloader:
                    
#                     audio_model.eval()
#                     input = i[0].to(device)
#                     target = i[1]

#                     for j in INSTRUMENTS:
#                         loss = criterion(audio_model(input,j),target[j].to(device))
                    
#                         total_loss += loss.item()
                
#                 avg_loss = total_loss / len(test_dataloader) / len(INSTRUMENTS)
#                 print("/tAverage loss during validation/test: {}".format(avg_loss))
#                 writer.add_scalar('Avergage_Loss/test', avg_loss, current_epoch+1)






# def fire_alarm_mechanism(temp_df,robust,fire_threshold):
#     fire_alarm_counter = 0
#     fire_raised_time = 0
#     if robust:
#         for i,row in temp_df.reset_index().iterrows():
#             if row['y_pred'] == 1:
#                 fire_alarm_counter += 1
#             else:
#                 fire_alarm_counter = 0
#             if fire_alarm_counter == fire_threshold:
#                 st.write('ALARM RAISED!')
#                 fire_raised_time = i
#                 break
#     else:
#         for i,row in temp_df.reset_index().iterrows():
#             if row['y_pred'] == 1:
#                 fire_alarm_counter += 1
#             else:
#                 if fire_alarm_counter > 0:
#                     fire_alarm_counter -= 1 # Reduce counter by 1
#             if fire_alarm_counter == fire_threshold:
#                 st.write('ALARM RAISED!')
#                 fire_raised_time = i
#                 break
#     return fire_raised_time

# def subplot_to_axes(vertical,horizontal):
#     """
#         Takes in the vertical and horizontal chart numbers and enumerates it for axes 
#         subplot_to_axes(3,2) => {1: [0, 0], 2: [0, 1], 3: [1, 0], 4: [1, 1], 5: [2, 0], 6: [2, 1]}
#     """
#     dic = {}
#     master_counter = 0
#     counter_i = -1
#     counter_j = -1
#     for i in range(vertical):
#         counter_i += 1
#         counter_j = -1
#         for j in range(horizontal):
#             counter_j += 1
#             master_counter += 1
#             dic[master_counter] = [counter_i,counter_j]
#     return dic

# def create_subplot(num_of_columns):
#     """
#         Takes in number of features and spits out the recommended length of chart
#     """
#     h = 2 # .floor always rounds down (round(3.8) == 3)
#     v = math.ceil(num_of_columns/h) # .ceil always rounds up (round(3.1) == 4)
#     return v,h

# def find_consecutive_missclassification(df, specified_threshold=5):
#     ''' 
#     Takes in a df and returns a dictionary of consecutive misclasses that are > than the specified threshold
#     '''
#     # st.write('time:', list(df['time']))
#     # st.write('binary misclass:',list(df['misclass']))
#     # st.write('truth label:',list(df['label']))
#     # st.write('prediction:', list(df['y_pred']))

#     dic = {}
#     one_counter = 0
#     for i in range(len(df)):
#         if df['misclass'].iloc[i] == 1:
#             one_counter += 1 
#         if df['misclass'].iloc[i] == 0 and one_counter >= specified_threshold:
#                 dic[i-1] = one_counter
#                 one_counter = 0 
#         if df['misclass'].iloc[i] == 0 and one_counter < specified_threshold:
#             one_counter = 0
#     lst = []
#     index_lst = []
#     for key,value in dic.items():
#         lst.append((df['time'].iloc[key-value+1],df['time'].iloc[key],value))
#         index_lst.append((key-value+1,key,value))
#     st.write("Format of ConsecMisclassification: (time_start, time_end, number of consecutive)")

#     return lst,index_lst


# def error_plot_analysis(selected_df,selected_test,selected_experiment,columns,robust,fire_threshold,specified_threshold=5,):
    
#     #Find the index of the first fire classification
#     index = (selected_df['label'].loc[(selected_df['test_type'] == selected_test) & (selected_df['experiment'] == selected_experiment)] != 0).argmax() # Finds the first occurance of true fire classification
#     time_of_fire = selected_df.loc[(selected_df['test_type'] == selected_test) & (selected_df['experiment'] == selected_experiment)].time.iloc[index]
    
# #     st.write(index) 
# #     st.write(time_of_fire)
    
#     if index != 0:
#         st.write("time of fire {}s({})".format(int(time_of_fire),index))
#         st.write("\n")
    
#     misclassification_size = 110

#     #Create custom pattern to color misclassified ambient(yellow) and fire(red) data
#     color = {}
#     colors_dic = {0:('yellow','Ambient'),1:('r','Fire'),2:('blue','Nuisance'),3:('green',"Flaming")}

#     for i,row in selected_df.loc[(selected_df['test_type'] == selected_test) & (selected_df['experiment'] == selected_experiment),['time','misclass','y_pred']].reset_index().iterrows():
#         if row['misclass'] == 1:
#             if row['y_pred'] == 0:
#                 color[np.float64(row['time'])] = 'yellow'
#             if row['y_pred'] == 1:
#                 color[np.float64(row['time'])] = 'r'
#             if row['y_pred'] == 2:
#                 color[np.float64(row['time'])] = 'blue'
#             if row['y_pred'] == 3:
#                 color[np.float64(row['time'])] = 'green'
    
#     # st.write("Sequence gap: {}, Number to be consec: {}".format(sequence_gap,number_to_be_consec))
#     st.subheader("ConsecMisclassification (Settings: >= {} Consec shown): \n".format(specified_threshold))
#     consecutive_misclass_lst , index_consecutive_misclass_lst = find_consecutive_missclassification(selected_df.loc[(selected_df['test_type'] == selected_test) & (selected_df['experiment'] == selected_experiment)], specified_threshold=specified_threshold)
#     st.markdown("Index of ConsecMisclassification: " + str(index_consecutive_misclass_lst))
#     st.markdown("Time of ConsecMisclassification: "+ str(consecutive_misclass_lst))
#     if Counter(color.values()) != Counter():
#         st.markdown(Counter(color.values()))

    
#     vertical, horizontal = create_subplot(len(columns))
    
#     axes_dict = subplot_to_axes(vertical, horizontal)
    
#     temp_df = selected_df.loc[(selected_df['test_type'] == selected_test) & (selected_df['experiment'] == selected_experiment),['time','y_pred']]
    
#     fire_raised_time = fire_alarm_mechanism(temp_df,robust,fire_threshold)

#     time_alarm_raised = 0   

#     if fire_raised_time != 0:
#         time_alarm_raised = selected_df.loc[(selected_df['test_type'] == selected_test) & (selected_df['experiment'] == selected_experiment),'time'].iloc[fire_raised_time]
#         if robust:
#             st.write("Robust alarm: Alarm rised at {}s({})".format(int(time_alarm_raised),fire_raised_time))
#         else:
#             st.write("Sensitive alarm: Alarm rised at {}s({})".format(int(time_alarm_raised),fire_raised_time))

#     if time_of_fire != selected_df.loc[(selected_df['test_type'] == selected_test) & (selected_df['experiment'] == selected_experiment),'time'].iloc[0] and index != 0:
#         if robust:
#             st.write("Time_difference {}s".format(int(time_alarm_raised-time_of_fire)))
#         else:
#             st.write("Time_difference {}s".format(int(time_alarm_raised-time_of_fire)))
    
#     fig, axes = plt.subplots(vertical, horizontal, figsize=(20, 20))
    
#     counter = 0
#     for column in columns:
#         counter += 1 
#         axes_label = axes_dict.get(counter)
#         df = selected_df.loc[(selected_df['test_type'] == selected_test) & (selected_df['experiment'] == selected_experiment),["y_pred","time","misclass",'test_type','experiment',column]]

#         feature = selected_df[column]
#         df.misclass.loc[df.misclass == 0] =  None
#         df.dropna(inplace=True)
#         df.misclass.loc[df.misclass == 1] =  feature + feature.std()/3
#         sns.lineplot(x="time", y=column,  data=selected_df.loc[(selected_df['test_type'] == selected_test) & (selected_df['experiment'] == selected_experiment),["time",'test_type','experiment',column]], ax=axes[axes_label[0], axes_label[1]])

#         if counter == 1: 
#             ax = sns.scatterplot(x="time", y="misclass",  data=df,hue="time", palette = color, ax=axes[axes_label[0], axes_label[1]], s = misclassification_size, legend=False)
#             handles = []
#             for i in df.y_pred.unique():
#                 col,label = colors_dic.get(i)
#                 handles.append(mpatches.Patch(color=col, label=label))
#             ax.legend(handles=handles, title = "Predicted Class:", 
#         fontsize = 'large', title_fontsize = "large")
#         else:
#             sns.scatterplot(x="time", y="misclass",  data=df,hue="time", palette = color, ax=axes[axes_label[0], axes_label[1]], s = misclassification_size, legend=False)
# #         axes[axes_label[0], axes_label[1]].set_ylim(np.min(selected_df[["time",column,'test_type','experiment']].loc[(selected_df['test_type'] == selected_test) & (selected_df['experiment'] == selected_experiment)][column]) - feature.std(), np.max(selected_df[["time",column,'test_type','experiment']].loc[(selected_df['test_type'] == selected_test) & (selected_df['experiment'] == selected_experiment)][column]) + feature.std())
    
#     for axe in axes: 
#         for i in axe:
#             if time_of_fire != selected_df.loc[(selected_df['test_type'] == selected_test) & (selected_df['experiment'] == selected_experiment),'time'].iloc[0] and index != 0:
#                 i.axvline(time_of_fire, ls='-', color = 'orange',label = 'Fire starts')
#             if fire_raised_time != 0:
#                 i.axvline(time_alarm_raised, ls='-', color = 'red',label = 'Alarm raised')
#             # i.axvline(400, ls='-', color = 'black', label = 'End of Experiment')

#     plt.tight_layout()
#     plt.legend()

#     st.pyplot(fig)

# class Validation:
#     def __init__(self, df_test, y_test, y_pred):
#         self.df_test = df_test
#         self.y_test = np.array(y_test)
#         self.y_pred = y_pred
    
#     def Misclassification(self):
#         misclassification_dict = {}
#         assert len(self.y_test) == len(self.y_pred)
#         for i in range(len(self.y_pred)):
#             if self.y_pred[i] != self.y_test[i]:
#                 misclassification_dict[i] = self.df_test['time'].iloc[i]
#         return misclassification_dict        


# class Features:
#     df_X = None
#     df_y = None
    
#     def __init__(self, df_raw, selected_features = None):
#         self.df_raw = df_raw
#         self.df_X_shape = None
#         self.df_y_shape = None
#         if selected_features != None:
#             self.selected_features = selected_features
#         else: 
#             self.selected_features = ['IR_grad', 'Blue_grad', 'IR_Blue_ratio', 'CO_grad', "label", "IR_integral", "Blue_integral", "CO_integral","IR_Blue_ratio_integral"]
        
#     def process(self):
#         df_X_selected = self.df_raw[self.selected_features].copy()
# #         df_X_selected.dropna(inplace=True)
#         df_y = df_X_selected["label"]
#         df_X_selected.drop(["label"], axis=1, inplace=True) 
#         self.df_X_shape = df_X_selected.shape
#         self.df_y_shape = df_y.shape
#         return df_X_selected, df_y

# st.title("SSY-603: Advanced Fire Detection")
# st.header("Input and configuration")
# st.subheader("Input files")
# user_input = {}
# user_input["input_csv"] = st.file_uploader(
#     "Model Data"
# )
# if user_input["input_csv"]:
#     zf = zipfile.ZipFile(user_input["input_csv"])
#     set_number_of_class = set()
#     set_test_device = set()
#     # set_experiment = set()

#     for fname in zf.infolist():
#         try:
#             if fname.filename not in ['model/','model/training_y_pred/','model/y_pred/','model/y_test/']:
#                 if 'model/training_y_pred/' in fname.filename[:22]:
#                         temp_string = fname.filename.replace('model/training_y_pred/','')
#                         set_number_of_class.add(temp_string[0])
#                         set_test_device.add(temp_string[13:17])
#                         # set_experiment.add(temp_string[18:-4])
#         except EOFError:
#             st.write("Folder was wrongly read")

#     num_of_class = st.selectbox(label='Number of class',options=sorted(list(set_number_of_class)))
#     test_device = st.selectbox(label='Test device',options=sorted(list(set_test_device)))

#     for fname in zf.infolist():
#         try:
#             if fname.filename not in ['model/','model/training_y_pred/','model/y_pred/']:
#                 if 'model/extracted_feature_data_label_v2.csv' in fname.filename:
#                         df_raw = pd.read_csv(zf.open(fname))
#                 if 'model/training_y_pred/' in fname.filename[:22]:
#                     if str(num_of_class) == fname.filename[22] and str(test_device) == fname.filename[13+22:17+22]:
#                         training_y_pred = joblib.load(zf.open(fname))
#                 if 'model/y_pred/' in fname.filename[:13]:
#                     if str(num_of_class) == fname.filename[13] and str(test_device) == fname.filename[13+13:17+13]:
#                         y_pred = joblib.load(zf.open(fname))
#                         break               
#         except EOFError:
#             st.write("Folder was wrongly read")

#     if num_of_class == '2':
#             df_raw.label.loc[df_raw.label == 2] = 0

#     df_raw = df_raw.loc[(df_raw['device'] != 'det1') | (df_raw['test_type'] != 'TF03') | (df_raw['experiment'] != 1039)]
#     df_raw = df_raw.loc[(df_raw['device'] != 'det2') | (df_raw['test_type'] != 'TF03') | (df_raw['experiment'] != 1029)]

#     df_raw = df_raw.loc[df_raw['time'] >= 0]
#     df_raw.dropna(inplace=True)

#     df_device_train = df_raw.loc[(df_raw["variant"] == "DOTC") & (df_raw["device"] != test_device)].copy()
#     df_device_test = df_raw.loc[(df_raw["variant"] == "DOTC") & (df_raw["device"] == test_device)].copy()
#     train = Features(df_device_train,selected_features = ['IR', 'Blue', 'T', 'CO','IR_grad', \
#                  'Blue_grad', 'T_grad', 'CO_grad', \
#                  'IR_Blue_ratio', 'T_integral', 'IR_Blue_ratio_integral', \
#                  'IR_Blue_ratio_grad','label','IR_Blue_sum', 'IR_Blue_mul'])

#     test = Features(df_device_test,selected_features = ['IR', 'Blue', 'T', 'CO','IR_grad', \
#                  'Blue_grad', 'T_grad', 'CO_grad', \
#                  'IR_Blue_ratio', 'T_integral', 'IR_Blue_ratio_integral', \
#                  'IR_Blue_ratio_grad','label','IR_Blue_sum', 'IR_Blue_mul'])
#     _, y_train = train.process()
#     X_test, y_test = test.process()

#     analysis = st.selectbox(label='Model Analysis',options=['Confusion Matrix','Error Plot Analysis'])

#     if analysis == 'Confusion Matrix':
#         st.header("Model Analysis")
#         st.header("Training")
#         st.subheader("Class: {} , Test Device: {}".format(num_of_class,test_device))

#         if num_of_class == '2': 
#             target_names = ["Non-Fire","Fire"]
#         if num_of_class == '3': 
#             target_names = ["Ambient","Fire","Nuisance"]

#         accuracy = accuracy_score(y_train, training_y_pred)
#         con = confusion_matrix(y_train, training_y_pred)
#         st.write("Confusion Matrix")

#         fig, ax = plt.subplots()
#         confusion_df = pd.DataFrame(con,columns=target_names)
#         confusion_df.index = target_names
#         ax = sns.heatmap(confusion_df, annot=True, fmt='d', cmap='PuBu')
#         plt.ylabel("Truth Class")
#         plt.xlabel("Predicted Class")
#         st.pyplot(fig)
#         fig, ax = plt.subplots()
#         st.write("Classification Report")

#         clf_report = classification_report(y_train, training_y_pred,target_names=target_names, output_dict=True)
#         ax = sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True, cmap='PuBu',vmin=0.90)
#         st.write("Accuracy: %.2f%%" % (accuracy * 100.0))

#         st.pyplot(fig)
#         st.write(pd.DataFrame(clf_report).iloc[:, :].T.style.format("{:.4f}"))

#         st.header("Test")

#         accuracy = accuracy_score(y_test, y_pred)
#         con = confusion_matrix(y_test, y_pred)
#         st.write("Confusion Matrix")

#         fig, ax = plt.subplots()
#         confusion_df = pd.DataFrame(con,columns=target_names)
#         confusion_df.index = target_names
#         ax = sns.heatmap(confusion_df, annot=True, fmt='d', cmap='PuBu')
#         plt.ylabel("Truth Class")
#         plt.xlabel("Predicted Class")
#         st.pyplot(fig)
#         fig, ax = plt.subplots()
#         st.write("Classification Report")

#         clf_report = classification_report(y_test, y_pred,target_names=target_names, output_dict=True)
#         ax = sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True, cmap='PuBu',vmin=0.90)
#         st.write("Accuracy: %.2f%%" % (accuracy * 100.0))

#         st.pyplot(fig)
#         st.write(pd.DataFrame(clf_report).iloc[:, :].T.style.format("{:.2f}"))
    
#     if analysis == 'Error Plot Analysis':
#         # VALIDATE MODEL
#         misclassification = Validation(df_device_test, y_test, y_pred).Misclassification()

#         selected_test = ""
#         df_device_test['y_pred'] = y_pred
#         df_device_test['misclass'] = 0
#         df_device_test['misclass_label'] = 0

#         for index, row in df_device_test[['time','misclass']].reset_index().iterrows():
#             if index in misclassification.keys():
#                 df_device_test['misclass'].iloc[index] = 1


#         selected_test_type = st.selectbox(label='Test type',options=list(df_device_test.loc[(df_device_test.device == test_device) & \
#                                                     (df_device_test.variant == "DOTC")].test_type.unique()))

#         selected_experiment = st.selectbox(label='Experiment',options=list(df_device_test.loc[(df_device_test.test_type == selected_test_type) & \
#                                                     (df_device_test.device == test_device) & \
#                                                     (df_device_test.variant == "DOTC")].experiment.unique()))


#         robust = st.checkbox(label='Robust Alarm')

#         fire_threshold = st.slider(label='Fire threshold', min_value=1, max_value=20,value = 15)

#         error_plot_analysis(df_device_test,selected_test_type,selected_experiment,X_test.columns,robust,fire_threshold,specified_threshold=5)
