import os
from pathlib import Path
path=str(Path.cwd())+"\outputs"
print ("path: ",path)
isExist = os.path.exists(path)
if not isExist:
   # Create a new directory because it does not exist
   os.makedirs(path)
   print("The new directory is created!")
else:
    print("The directory Exists!")
algorithm=["bfs","dfs","ucs","astar,heuristic=manhattanHeuristic","bds_mm,heuristic=manhattanHeuristic","bds_mm0"]
mazes=["contoursMaze","openMaze", "smallMaze","mediumMaze"]
# for each maze in mazes
k=0
for j in range(len(mazes)):
    for i in range(len(algorithm)):
        file=""
        if "," in algorithm[i]:
            index=algorithm[i].index(",")
            file=algorithm[i][:index]
        else:
            file=algorithm[i]
        s="python pacman.py -l "+mazes[j]+" -z .5 -p SearchAgent -a fn="+algorithm[i]+" --frameTime 0 >./outputs/output"+str("{0:0=3d}".format(k))+"_"+mazes[j]+"_"+file+".txt"
        print ("command executing: ",s)
        os.system(s)
        k+=1

# for big mazes
for j in range(1,16):
    for i in range(len(algorithm)):
        file=""
        if "," in algorithm[i]:
            index=algorithm[i].index(",")
            file=algorithm[i][:index]
        else:
            file=algorithm[i]
        s="python pacman.py -l bigMaze"+str(j)+" -z .5 -p SearchAgent -a fn="+algorithm[i]+" --frameTime 0  >./outputs/output"+str("{0:0=3d}".format(k))+"_bigMaze"+str(j)+"_"+file+".txt"
        print ("command executing: ",s)
        os.system(s)
        k+=1

# reading and writing all files into result file
read_files = os.listdir(path)
print (read_files)
open(path+"\\"+'result.txt', 'w').close()
with open(path+"\\"+"result.txt", "wb") as outfile:
    for f in read_files:
        f=path+"\\"+f
        with open(f, "rb") as infile:
            print("file: ",f)
            if "_" in f:
                s=f[f.rindex("_",0,f.rindex("_")-1)+1:f.rindex(".")]+"\n"
                outfile.write(s.encode())
                outfile.write(infile.read())
                outfile.write("\n".encode())