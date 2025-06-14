#import torchaudio as ta
#from chatterbox.tts import ChatterboxTTS
import sys
import re
import os

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

BATCH_SIZE=249
WORDS_LEN=len(words)
N_BATCH_SIZE_JOBS=round(WORDS_LEN/BATCH_SIZE)

jobs = []

if len(words) > BATCH_SIZE:
  for i in range(N_BATCH_SIZE_JOBS):
    start=i*BATCH_SIZE
    end=(i*BATCH_SIZE)+BATCH_SIZE
    jobs.append(" ".join(words[start:end]));

  if N_BATCH_SIZE_JOBS is not WORDS_LEN / BATCH_SIZE:
    start=N_BATCH_SIZE_JOBS*BATCH_SIZE
    end=WORDS_LEN
    jobs.append(" ".join(words[start:end]));
    assert(len(jobs) == N_BATCH_SIZE_JOBS + 1);

  else:  assert(len(jobs) == N_BATCH_SIZE_JOBS);

else: jobs.append(words);

# make the output dir accept it as one of the command line input
TMPDIR="/tmp/tts_gen_6940";
try: os.mkdir(TMPDIR);
except:
  try: os.rmdir(TMPDIR);
  except:
    print(f"failed to rmdir: {TMPDIR}");
    exit(1);

  try: os.mkdir(TMPDIR);
  except:
    print(f"failed to mkdir: {TMPDIR}");
    exit(0);

print(f"successfully created clean dir {TMPDIR}");
exit(0)

for i in range(len(jobs)):
  model = ChatterboxTTS.from_pretrained(device="cpu");
  wav = model.generate(" ".join(job[i]));
  ta.save(wavfilepath, wav, model.sr);

file.close();
exit(0);
