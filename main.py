#import torchaudio as ta
#from chatterbox.tts import ChatterboxTTS
import sys
import re

textfilepath="";
wavfilepath="";
content="";

def usage():
  print("python3 main.py [text filepath] [output .wav filepath]");

try: 
  textfilepath = sys.argv[1];
  wavfilepath = sys.argv[2];
except:
  usage();
  exit(1);

try: 
  file = open(textfilepath);
except:
  print("failed to open file: {textfilepath}");
  exit(1);

try:
  content = file.read();
except:
  print("failed to read file: {textfilepath}");
  exit(1);

# do something that will limit batch text to something like 250 words or whatever
# arr = re.split("[?.!]", content);
words = re.split(" ", content);

#print(len(words));

BATCH_SIZE=249
WORDS_LEN=len(words)
N_BATCH_SIZE_JOBS=round(WORDS_LEN/BATCH_SIZE)

jobs = []
for i in range(N_BATCH_SIZE_JOBS):
  start=i*BATCH_SIZE
  end=(i*BATCH_SIZE)+BATCH_SIZE
  #print(f"{WORDS_LEN}>{start}:{end}::{end - start}");
  #print(" ".join(words[start:end]));
  jobs.append(" ".join(words[start:end]));

# empty out the remaining bit that is less that a full batch
# but has still not been processed
if N_BATCH_SIZE_JOBS is not WORDS_LEN / BATCH_SIZE:
  start=N_BATCH_SIZE_JOBS*BATCH_SIZE
  end=WORDS_LEN
  #print(f"{WORDS_LEN}>{start}:{end}::{end - start}");
  #print(" ".join(words[start:end]));
  jobs.append(" ".join(words[start:end]));
  assert(len(jobs) == N_BATCH_SIZE_JOBS + 1);

else:  assert(len(jobs) == N_BATCH_SIZE_JOBS);

#model = ChatterboxTTS.from_pretrained(device="cpu");
#wav = model.generate(" ".join(words[:249]));
#ta.save(wavfilepath, wav, model.sr);

file.close();
exit(0);
