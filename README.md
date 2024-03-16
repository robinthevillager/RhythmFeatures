## Rhythm Feature Extraction Pipeline
#### MIR - Course 2023/2024 by Martín Rocamora - UPF Barcelona - Sound and Music Computing
#### Authors: Robin Doerfler, Anmol Mishra, Satyajeet Prabhu

### Description:
This repository contains a feature extraction pipeline for extracting rhythm features from audio files. 
The extracted features include tempo, downbeats, beats, micro-timings, and onset loudnesses.
For every file of the dataset, the we create a corresponding folder in the `output` folder.
This folder contains the following files:
- `File_Name.mid` : A midi file containing ... 
  - Notes (C3) at every onset position
  - The velocity of the notes corresponding to the loudness of the onset
  - The tempo as BPM value in the header of the midi file
- `File_Name.json` : A json file containing the extracted rhythm features as time series
  - `tempo` : The tempo of the audio file in BPM
  - `downbeat_times` : The downbeat positions in seconds
  - `beat_times` : The beat positions in seconds
  - `microtiming` : The microtiming deviations from the grid in seconds
  - `onset_loudness` : The loudness of the onsets (A-weighted and normalised spectral power of note)
- `File_Name_onset_sonification.wav` : The original audio file with superimposed click-sounds at the onset positions with the corresponding loudness
- `File_Name_rhythm_features.png` : A plot of the extracted rhythm features including the tempo, downbeats, beats, and onset loudnesses

### How to use:
1. Clone the repository
2. Install the required packages using `pip install -r requirements.txt`
3. Copy the audio files of the dataset into the audiofiles/drumloops folder
4. Run the `main.py` file using `python main.py`
5. The extracted features will be saved in the `output` folder


### Structure:
- `audiofiles` : Contains the audio files of the dataset
- `output` : Contains the extracted rhythm features for every audio file of the dataset
- `src` : Contains the source code of the feature extraction pipeline
  - `main.py` : The main file to run the feature extraction pipeline
  - `featureprocessor.py` : Contains the main feature processor class
  - `dsp_functions.py`: Contains supplementary dsp functions
  - `midi_functions.py` : Contains supplementary midi functions
  - `plotting_functions.py` : Contains supplementary plotting functions
  - `utils.py` : Contains various utility functions

### Remarks:

The current state of the pipeline assumes a dataset of drumloops which has the following constraints:
- The audio files are sliced as 'loops' which repeat after an integer number of bars
- The tempo is annotated in the file name as integer number of beats per minute and is the only number in the file name


### Background:

#### Motivation:
Styles of Latin American (ex.Samba) and African music (ex.Chaabi) have their own unique rhythmic swing that defines their identity. Certain modern genres like Lo-Fi hip hop and artists like J Dilla and Burial incorporate a distinctive swing into their work, creating a dynamic and unique groove with varying amounts of swing and accentuation across different tempos and musical contexts. For musicians and producers, it can be challenging to incorporate these rhythmic signatures, especially when relying on workflows for drum sequence programming based on a quantization grid.

Current digital audio workstations, such as Ableton Live, offer static 'groove patterns' that apply fixed micro-timings and velocity offsets to MIDI sequences. [1] However, these predetermined patterns may not fully capture the complexity and nuance of a human performer's rhythmic style. A simple stochastic system, while capable of generating drum patterns based on observed probabilities, may fail to convey the subtle variations and "meaning" expressed by a musician playing the same rhythm for instance at different tempos.[2]

To address this issue, developing a machine learning system that transfers a desired rhythmic style, such as Bossa Nova, onto a quantized MIDI clip could prove useful. Given a quantized MIDI sequence of a drum rhythm, this system would predict microtiming corrections and specify beat emphasis according to the chosen musical style. 

To facilitate the creation of such a machine learning algorithm and ensure its adaptability to various musical styles, setting up a pipeline that generates suitable feature sets directly from unannotated audio clips is essential. These clips can include self-recorded samples, loop libraries, or any other relevant sources.

#### Scope of Project:

The scope of this project lies specifically within the development and testing of the feature extractor pipeline. Our goal is to implement the pipeline using a curated set of audio loops showcasing the bossa nova style. We chose this underrepresented genre due to the limited availability of existing datasets and the potential benefits of applying our methodology to other musical styles lacking extensive data resources.

Working with a collection of audio recordings featuring a solo drummer, we will extract both the tempo and the drum groove information from the audiotracks, including micro-timings and onset strengths for individual hits. Because the limited amount of recordings were mainly performed by a single artist, they represent a common starting point for researchers working with less popular or underexplored music genres.

Previous research indicates that satisfactory results can be achieved with relatively small datasets when addressing specialized machine learning tasks. For instance, a meter tracking model for Latin American music produced acceptable outcomes after being trained on only three minutes of audio material supplemented with automated augmentation techniques. [3]

Our current project focuses exclusively on establishing the feature extraction pipeline; however, future endeavors should explore and create the complementary machine learning algorithms necessary for estimating micro-timings and accentuations based on quantized and sterile MIDI drum patterns generated via user input or stochastic methods. By building upon our foundation, subsequent studies can expand on our findings and enhance the accuracy and applicability of these tools in diverse musical settings.

#### Dataset / Resources:

For our approach we want to take a simple and small dataset only consisting of audio files and their global tempo annotation. Based on this, we would estimate downbeats, beats & tempo using existing implementations of state-of-the-art models and toolboxes. After deriving this quantisation grid, we would like to calculate the microtiming deviations from the grid, i.e. for every 16th note, as well as onset strength and potentially other features to provide meaningful context for later training of machine learning algorithms.


We foresee to make use of the following resources:

Implementations & code of existing models and toolboxes:
Madmom (Downbeat & Beat Estimation) [4]
Carat Toolbox [5]
Librosa [6]
Data
Elevador: Brazilian Bossa Nova (Sample Pack by Soul Surplus on Splice.com) [7]
Baseline:
The Filosax dataset [8] annotates the swing amount for jazz music for which they use a similar approach of extracting tempo followed by calculating swing ratios. They, however, use the proprietary DAW Logic for pre-processing their data. We aim to build a pipeline for our approach using existing open-source tools and research for extracting tempo, downbeat and swing information.
Also, our idea is to annotate rhythmic groove in a way to make it useable for machine learning.


#### References / Papers:

[1] https://www.ableton.com/en/manual/using-grooves/

[2] Guilherme Schmidt Câmara: Timing Is Everything . . . Or Is It?
Investigating Timing and Sound Interactions in the Performance of Groove-Based Microrhythm

[3] Maia, L. S., Rocamora, M., Biscainho, L. W. P., & Fuentes, M. (2022). Adapting Meter Tracking Models to Latin American Music. Accepted at ISMIR 2022. arXiv:2304.07186 [cs.SD]

[4] S. Böck, F. Krebs and G. Widmer: Joint Beat and Downbeat Tracking with Recurrent Neural Networks, Proceedings of the 17th International Society for Music Information Retrieval Conference (ISMIR), 2016.

[5] Rocamora, and Jure. "carat: Computer-Aided Rhythmic Analysis Toolbox." In Proceedings of Analytical Approaches to World Music. 2019.

[6] Brian McFee, Colin Raffel, Dawen Liang, Daniel Ellis, Matt Mcvicar, Eric Battenberg, Oriol Nieto. “Librosa: Audio and Music Signal Analysis in Python.” Python in Science Conference 2015.

[7] Elevador: Brazilian Bossa Nova (Sample Pack by Soul Surplus on Splice.com)
https://splice.com/sounds/packs/soul-surplus/elevador-brazilian-bossa-nova/samples?tags=8db1fa7d-2ac6-4fae-9f97-fdb13690135b&asset_category_slug=loop

[8] D. Foster and S. Dixon (2021), Filosax: A Dataset of Annotated Jazz Saxophone Recordings. 22nd International Society for Music Information Retrieval Conference (ISMIR).

