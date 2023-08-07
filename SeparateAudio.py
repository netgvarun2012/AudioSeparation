from pyannote.audio import Pipeline
from IPython.core.display import display
import os 
import IPython                    # displaying audio
from scipy.io import wavfile
import scipy.io                   # scientific data handling
import numpy as np
import matplotlib.pyplot as plt   # visualization framework
import seaborn as sns             # additional visualization support
import glob
from pydub import AudioSegment
from pydiarization.diarization_wrapper import rttm_to_string
import argparse
import time

def readRTTM(file):
    data = {}
    rttm = rttm_to_string(file)
    rttm = rttm.split('\n')[:-1]
    for line in rttm:
        data[float(line.split(' ')[3])] = [float(line.split(' ')[4]),
                                           line.split(' ')[7]]
    starts, intervals, speakers = [], [], []
    for start in sorted(data): # sort returns by start
        starts.append(start)
        intervals.append(data[start][0])
        speakers.append(data[start][1])
    return starts, intervals, speakers

def separateAudio(starts,intervals,speakers,samplerate,audioPath,inputfile,data,speaker_num):
  for annotationIndex in range(len(starts)):    
    start = np.round(starts[annotationIndex], 2)
    end = np.round(starts[annotationIndex] + intervals[annotationIndex], 2)
    print(f'\n\n [{start} - {end}] {speakers[annotationIndex]}\n')
    arrStart = start * samplerate
    arrEnd = end * samplerate
    arrStart, arrEnd = int(arrStart), int(arrEnd)
    scipy.io.wavfile.write('temp_'+str(annotationIndex)+'.wav', samplerate, data[arrStart:arrEnd])
    #display(IPython.display.Audio('temp_'+str(annotationIndex)+'.wav'))

  tempwavfiles = sorted(glob.glob('temp_*.wav'))
  silence_duration = 5 * 100 # 0.5 seconds (or 500 milliseconds)
  silenced_segment = AudioSegment.silent(duration=silence_duration)        
  
  # Iterate over the list of files and process/remove individually
  filenamescombined = []
  #print(f'tempwavfiles is {tempwavfiles}')
  combined = AudioSegment.empty()
  for file in tempwavfiles:
    #print(f'file is {file}')
    audiofilename = AudioSegment.from_wav(file)
    #print(f'audiofilename is {audiofilename}')
    filenamescombined.extend([audiofilename,silenced_segment])
    os.remove(file)

  #print(f'filenamescombined is {filenamescombined}')
  for fname in filenamescombined:
  #print(f'fname is {fname}')
    combined += fname

  generatedFile = inputfile+"_"+str(speaker_num)+".wav"
  combined.export(audioPath + generatedFile, format="wav")
  print(f'Final wav recording Generated For Speaker {speaker_num} \n')
  #display(IPython.display.Audio('/content/drive/MyDrive/AudioSeparation/combined_new_file.wav'))


def exploreFile(num_speakers,
                fileinput,
                playAudio = True,
                displayIntervals = True,
                audioPath = '/content/drive/MyDrive/AudioSeparation/',
                rttmPath = '/content/drive/MyDrive/AudioSeparation'
                ):
    
    # read in data
    #file = glob.glob(audioPath+'/*.wav',recursive=True)[index].split(os.path.sep)[-1] #os.listdir(audioPath)[index]
    file = fileinput
    print(f'file is {file}')
    samplerate, data = wavfile.read(os.path.join(audioPath, file))
    starts, intervals, speakers = readRTTM(os.path.join(rttmPath, file.split('.')[0] + '.rttm'))
    print(f'starts is {starts}')
    print(f'intervals is {intervals}')
    print(f'speakers is {speakers}')
   
    if num_speakers:
        if num_speakers != len(set(speakers)):
            print('\nrttm file is corrputed. Improper diarization . Exiting... !\n')
            exit(0)
    else:
        num_speakers = len(set(speakers))
        
    # separating the speakers details into a dictionary
    speaker_data = {}
    for i in range(num_speakers):
        if speakers[i] not in speaker_data:
            speaker_data[speakers[i]] = {
                'starts': [],
                'intervals': [],
                'speakers': []
            }
        speaker_data[speakers[i]]['starts'].append(starts[i])
        speaker_data[speakers[i]]['intervals'].append(intervals[i])
        speaker_data[speakers[i]]['speakers'].append(speakers[i])    
    
   

    # play audio file and associated intervals
    if playAudio:
        
        print('COMPLETE AUDIO FILE\n')
        scipy.io.wavfile.write('temp.wav', samplerate, data)
        #display(IPython.display.Audio('temp.wav'))
        
        print('-'*50)

        print('SAMPLED SPEAKING INTERVALS')

    print(f'speaker_data items() is {speaker_data.items()}\n')
    # Iterate over each speaker's data and call separateAudio
    for idx,(speaker, S_data) in enumerate(speaker_data.items()):
        print(f'idx is {idx}\n')
        separateAudio(S_data['starts'], S_data['intervals'], S_data['speakers'], samplerate, audioPath, file, data, idx)

def get_file_name(link):
    newPath = link.replace(os.sep, '/')
    filename = newPath.split('/')[::-1][0]
    return (filename.split('.')[::-1][1])

def get_file_path(link):
    """
    This function splits the filename and returns path without the name

    :param file: full filename with extension

    :return: path without filename
    """     

    return os.path.dirname(link)


def main():
    """
    This is the main function for separating audio files based on different speakers.
    """
    parser = argparse.ArgumentParser(description="SeparateAudio", usage='%(prog)s [-h] [-w wavfile] [-n NumberOfSpeakers]')

    parser.add_argument("-w", "--wavfile", type=str, help="Input wavfile to be separated (absolute path)", required=False)
    parser.add_argument("-n", "--NumberOfSpeakers", type=int, help="Number of speakers to separate", required=False)

    args = parser.parse_args()
    wavfile = args.wavfile
    num_speakers = args.NumberOfSpeakers

    #if not wavfile or not num_speakers:
    if not wavfile:
        print("'-w'argument is missing!")
        print()
        parser.print_usage()
        print()
        exit(0)

    filewoextension = get_file_name(wavfile)
    filepath = get_file_path(wavfile)
    filepath +="/"

    print(f'filepath is {filepath}')
    if wavfile:
        print('\nApplying pipeline to the audio file....')
        # apply the pipeline to an audio file

        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",
                                    use_auth_token="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")

        if num_speakers: # number of speakers are given as parameter
            diarization = pipeline(wavfile,num_speakers=num_speakers)
        else:
            diarization = pipeline(wavfile)
        print('\nDumping the diarization output to disk using RTTM format....')

        # dump the diarization output to disk using RTTM format
        with open(filepath+filewoextension+".rttm", "w") as rttm:
            diarization.write_rttm(rttm)
        print('\nCalling the explore file to split into multiple speakers as per RTTM format....')
        exploreFile(num_speakers,filewoextension+'.wav',     audioPath = filepath,
                rttmPath = filepath)

if __name__ == "__main__":
    start_time = time.time()  # Record the start time
    main()
    end_time = time.time()  # Record the end time
    # Calculate the execution time
    execution_time_minutes = (end_time - start_time) / 60

    print(f"Execution time: {execution_time_minutes:.2f} minutes")

