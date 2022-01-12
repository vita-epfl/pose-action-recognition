#!/bin/bash
#SBATCH --nodes 1
#SBATCH --cpus-per-task 2
#SBATCH --ntasks=8
#SBATCH --account=vita
#SBATCH --mem-per-cpu=4G
#SBATCH --time 72:00:00
#SBATCH --gres=gpu:1
#SBATCH --output "slurm-%A_%a.log"

echo "starting to process CASR dataset"

cd ./scratch

git clone https://github.com/VRU-intention/casr.git

cd ./scratch/casr

mkdir data
unzip annotations.zip -d data
cd data
mkdir videos 
cd ..

echo "Downloading first dataset" 
wget -O person01_style1.zip http://adas.cvc.uab.es/casr/download/person01_style1/?wpdmdl=21&refresh=60a3a6c7e22251621337799 
echo "Downloading second dataset" 
wget -O person01_style2-3.zip http://adas.cvc.uab.es/casr/download/person01_style2-3/?wpdmdl=23&refresh=60a3a6f8b5f301621337848 
echo "Downloading third dataset" 
wget -O person01_style4.zip http://adas.cvc.uab.es/casr/download/person01_style4/?wpdmdl=24&refresh=60a3a70a9dc6d1621337866 
echo "Downloading forth dataset" 
wget -O person01_style5-6.zip http://adas.cvc.uab.es/casr/download/person01_style5-6/?wpdmdl=26&refresh=60a3a7208a1101621337888 
echo "Downloading fifth dataset" 
wget -O person02_style1.zip http://adas.cvc.uab.es/casr/download/person02_style1/?wpdmdl=27&refresh=60a3a73155ff11621337905 
echo "Downloading sixth dataset" 
wget -O person02_style2-3.zip http://adas.cvc.uab.es/casr/download/person02_style2-3/?wpdmdl=28&refresh=60a3a740a372e1621337920 
echo "Downloading seventh dataset" 
wget -O person03_style1-2.zip http://adas.cvc.uab.es/casr/download/person03_style1-2/?wpdmdl=29&refresh=60a3a7540aff91621337940 
echo "Downloading eighth dataset" 
wget -O person04_styles.zip http://adas.cvc.uab.es/casr/download/person04_styles/?wpdmdl=30&refresh=60a3a764ea5c81621337956 
echo "Downloading youtube videos" 
wget -O youtuber.zip http://datasets.cvc.uab.es/CASR/youtuber.zip
wait
echo "Finish downloading datasets"

echo "Unzip videos..." 
unzip person01_style1.zip -d data/videos 
unzip person01_style2-3.zip -d data/videos 
unzip person01_style4.zip -d data/videos 
unzip person01_style5-6.zip -d data/videos 
unzip person02_style1.zip -d data/videos 
unzip person02_style2-3.zip -d data/videos 
unzip person03_style1-2.zip -d data/videos 
unzip person04_styles.zip -d data/videos 
unzip youtuber.zip -d data/videos 
echo "Done unzipping" 

echo "Exctracting images" 
sh extract_images.sh 

echo "Done extracting images" 

module load gcc/8.4.0 python/2.7.18

cd ./scratch/casr

python -m virtualenv venv 

source ./venv/bin/activate 

python -m pip install --user joblib opencv-python==3.4.3.18

for i in {0..178} 
do
	{
	echo "Ground truth for $i"
	python2 view.py --dataset casr --track_number $i
	} &
done
wait
echo "Done making ground truth" 
