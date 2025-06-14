import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
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

model = ChatterboxTTS.from_pretrained(device="cpu");
wav = model.generate(" ".join(words[:249]));
ta.save(wavfilepath, wav, model.sr);

file.close();
exit(0);
