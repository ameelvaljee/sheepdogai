import pickle
import math
import util.mapelites as mapelites
import os
import statistics as stat
import matplotlib.pyplot as plt
import process.project_archive as prja
from util.config_reader import ConfigReader

def mean_flatten(array):
  output = []
  for i in range(len(array)):
    output.append(stat.mean(array[i]))
  return output

def graph(variant, runs=20, generations=150):

  MAX_RUNS = runs
  if variant == "hom":
    AGGREGATE_PREFIXES = ["shom-e", "shom-m", "shom-d", "mhom-e", "mhom-m", "mhom-d"]
  elif variant == "het":
    AGGREGATE_PREFIXES = ["shet-e", "shet-m", "shet-d", "mhet-e", "mhet-m", "mhet-d"]
  elif variant == "ahet":
    AGGREGATE_PREFIXES = ["ashet-e", "ashet-m", "ashet-d", "amhet-e", "amhet-m", "amhet-d"]
  elif variant == "hom-e":
    AGGREGATE_PREFIXES = ["shom-e-e", "shom-m-e", "shom-d-e", "mhom-e-e", "mhom-m-e", "mhom-d-e"]
  elif variant == "het-e":
    AGGREGATE_PREFIXES = ["shet-e-e", "shet-m-e", "shet-d-e", "mhet-e-e", "mhet-m-e", "mhet-d-e"]
  elif variant == "ahet-e":
    AGGREGATE_PREFIXES = ["ashet-e-e", "ashet-m-e", "ashet-d-e", "amhet-e-e", "amhet-m-e", "amhet-d-e"]
  elif variant == "hom-m":
    AGGREGATE_PREFIXES = ["shom-e-m", "shom-m-m", "shom-d-m", "mhom-e-m", "mhom-m-m", "mhom-d-m"]
  elif variant == "het-m":
    AGGREGATE_PREFIXES = ["shet-e-m", "shet-m-m", "shet-d-m", "mhet-e-m", "mhet-m-m", "mhet-d-m"]
  elif variant == "ahet-m":
    AGGREGATE_PREFIXES = ["ashet-e-m", "ashet-m-m", "ashet-d-m", "amhet-e-m", "amhet-m-m", "amhet-d-m"]
  elif variant == "hom-d":
    AGGREGATE_PREFIXES = ["shom-e-d", "shom-m-d", "shom-d-d", "mhom-e-d", "mhom-m-d", "mhom-d-d"]
  elif variant == "het-d":
    AGGREGATE_PREFIXES = ["shet-e-d", "shet-m-d", "shet-d-d", "mhet-e-d", "mhet-m-d", "mhet-d-d"]
  elif variant == "ahet-d":
    AGGREGATE_PREFIXES = ["ashet-e-d", "ashet-m-d", "ashet-d-d", "amhet-e-d", "amhet-m-d", "amhet-d-d"]

  elif variant == "easy-het":
    AGGREGATE_PREFIXES = ["shet-e", "shet-e-e", "shet-e-m", "shet-e-d", "mhet-e", "mhet-e-e", "mhet-e-m", "mhet-e-d"]
  elif variant == "med-het":
    AGGREGATE_PREFIXES = ["shet-m", "shet-m-e", "shet-m-m", "shet-m-d", "mhet-m", "mhet-m-e", "mhet-m-m", "mhet-m-d"]
  elif variant == "diff-het":
    AGGREGATE_PREFIXES = ["shet-d", "shet-d-e", "shet-d-m", "shet-d-d", "mhet-d", "mhet-d-e", "mhet-d-m", "mhet-d-d"]
  elif variant == "easy-hom":
    AGGREGATE_PREFIXES = ["shom-e", "shom-e-e", "shom-e-m", "shom-e-d", "mhom-e", "mhom-e-e", "mhom-e-m", "mhom-e-d"]
  elif variant == "med-hom":
    AGGREGATE_PREFIXES = ["shom-m", "shom-m-e", "shom-m-m", "shom-m-d", "mhom-m", "mhom-m-e", "mhom-m-m", "mhom-m-d"]
  elif variant == "diff-hom":
    AGGREGATE_PREFIXES = ["shom-d", "shom-d-e", "shom-d-m", "shom-d-d", "mhom-d", "mhom-d-e", "mhom-d-m", "mhom-d-d"]

  
  elif variant == "easy-ahet":
    AGGREGATE_PREFIXES = ["ashet-e", "ashet-e-e", "ashet-e-m", "ashet-e-d", "amhet-e", "amhet-e-e", "amhet-e-m", "amhet-e-d"]
  elif variant == "med-ahet":
    AGGREGATE_PREFIXES = ["ashet-m", "ashet-m-e", "ashet-m-m", "ashet-m-d", "amhet-m", "amhet-m-e", "amhet-m-m", "amhet-m-d"]
  elif variant == "diff-ahet":
    AGGREGATE_PREFIXES = ["ashet-d", "ashet-d-e", "ashet-d-m", "ashet-d-d", "amhet-d", "amhet-d-e", "amhet-d-m", "amhet-d-d"]
  
  


  for prefix in AGGREGATE_PREFIXES:
    folders = [("output/" + folder) for folder in os.listdir("output") if folder.startswith("run_" + prefix)]

    if len(folders) > MAX_RUNS:
      folders = folders[:MAX_RUNS]

    AGGREGATE_ARRAY = None

    folder_count = 0

    for folder in folders:
      if os.path.exists(folder + "/checkpoints/gen_" + str(generations) + ".pkl"):
        flag = AGGREGATE_ARRAY is None
        if flag:
          AGGREGATE_ARRAY = []
        for i in range(1, generations+1):
          if flag:
            AGGREGATE_ARRAY.append([])
          CHECKPOINT_FILENAME = folder + "/checkpoints/gen_" + str(i) + ".pkl"
          if "shom" in folder or "shet" in folder or "amhet" in folder:
            grid = prja.project(CHECKPOINT_FILENAME)
            fitness_grid = grid.quality_array
          else:
            with open(CHECKPOINT_FILENAME, "rb") as cp_file:
              CHECKPOINT = pickle.load(cp_file)
            POPULATION = CHECKPOINT["pop"]
            CONFIG_FILENAME = CHECKPOINT["cfg"]
            CONFIG = ConfigReader(CONFIG_FILENAME)
            mapelites.init(CONFIG.get("pBehaviourFeatures", "[str]"), POPULATION)
            
            fitness_grid = mapelites.grid.quality_array
          total = 0
          qd_score = 0
          for x in range(len(fitness_grid)):
            for y in range(len(fitness_grid[x])):
              for z in range(len(fitness_grid[x][y])):
                if not math.isnan(fitness_grid[x][y][z]):
                  fitness = fitness_grid[x][y][z][0]
                  qd_score += fitness
                  total += 1

          # Normalize qd_score
          normalized_qd_score = qd_score / total 

          # Append to the aggregate array
          AGGREGATE_ARRAY[i-1].append(normalized_qd_score)

        folder_count += 1
      else:
        print("Skipping run: " + (folder + "/checkpoints/gen_" + str(generations) + ".pkl") + " is missing.")

    AGGREGATE_ARRAY = mean_flatten(AGGREGATE_ARRAY)

    style = "-" if "sh" in prefix else "--"

    if "mhom" in prefix:
      p = "MHOM"
    elif "shom" in prefix:
      p = "SHOM"
    if variant in ["het", "hom", "ahet", "het-diffmap", "hom-diffmap", "ahet-diffmap"]:
      if "-e" in prefix:
        color = "g"
      elif "-m" in prefix:
        color = "b"
      elif "-d" in prefix:
        color = "r"

    elif variant in [ "het-easymap", "hom-easymap", "ahet-easymap"]:
      if "-e-e" in prefix:
        color = "g"
      elif "-m-e" in prefix:
        color = "b"
      elif "-d-e" in prefix:
        color = "r"

    elif variant in [ "het-medmap", "hom-medmap", "ahet-medmap"]:
      if "-e-m" in prefix:
        color = "g"
      elif "-m-m" in prefix:
        color = "b"
      elif "-d-m" in prefix:
        color = "r"

    elif variant in [ "easy-het", "easy-hom", "easy-ahet"]:
      if "-e-e" in prefix:
        color = "g"
        p = p+ "-E-Simple maze"
      elif "-e-m" in prefix:
        color = "b"
        p = p+ "-E-Medium maze"
      elif "-e-d" in prefix:
        color = "r"
        p = p+ "-E-Difficult maze"
      elif "-e" in prefix:
        color = "c"
        p = p+ "-E-No maze"
      

    elif variant in [ "med-het", "med-hom", "med-ahet"]:
      if "-m-e" in prefix:
        color = "g"
        p = p+ "-D-Simple maze"
      elif "-m-m" in prefix:
        color = "b"
        p = p+ "-D-Medium maze"
      elif "-m-d" in prefix:
        color = "r"
        p = p+ "-D-Difficult maze"
      elif "-m" in prefix:
        color = "c"
        p = p+ "-D-No maze"
        
    elif variant in [ "diff-het", "diff-hom", "diff-ahet"]:
      if "-d-e" in prefix:
        color = "g"
      elif "-d-m" in prefix:
        color = "b"
      elif "-d-d" in prefix:
        color = "r"
      elif "-d" in prefix:
        color = "c"


    plt.plot(AGGREGATE_ARRAY, label=p, c=color, ls=style)

    print("Results plotted for " + str(folder_count) + " run(s).")

  #title = "Homogeneous" if variant == "hom" else "Heterogeneous"

  #plt.suptitle(title, weight="bold")
  #plt.title("SSGA vs. MAP-Elites", fontsize=10)
  plt.xlabel("Generation")
  plt.ylabel("Average QD score")
  plt.ylim(top=1.0)
  plt.legend(loc="upper left")
  plt.savefig("output/qdscore-" + variant + ".png", bbox_inches='tight', pad_inches=0.2)