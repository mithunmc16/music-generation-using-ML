"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

This code is written as part of COMP8755- Individual Computing Project

This code is for preprocessing and training the LSTM network

The songs are first parsed and the notes of different instruments are procesed to train and mdh5 files for music generation.


@Author:
Mithun
u6849970

@ Supervisor:
Professor Nick Birbillis

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


from music21 import *
import glob
import pickle
import numpy
"""
Keras is the library which we will use to train the model.
"""
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint


def get_notes():
    """
This Function is to get the midi file from the dataset and to extract the notes, rests and chords
"""
    notes = []  
    rest = True
    for file in glob.glob("midi\*.mid"):
	
        """
	we loop through various midi fils in the directory and parses them using M21 library
	
	"""
        midi = converter.parse(file)
        print("Parsing %s" % file)
        notes_to_parse = None
        """
		We extract the number of instruments in each MIDI file
        
"""
        try:  
            inst = instrument.partitionByInstrument(midi)
            print("Number of instrument parts: " + str(len(inst.parts)))
            notes_to_parse = inst.parts[0].recurse()
            """
		The exception condition here is for the rests
"""		
        except:  
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            """
        For each elements in the notes we check if it is a chords, or a note or a rest. 
        
        and then store it in a file
		"""
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
            elif isinstance(element, note.Rest) and rest:
                notes.append("rest")
        """
	Adding notes to the file notes
	
	"""
    with open('data\music', 'wb') as filepath:
        pickle.dump(notes, filepath)

    
    return notes

"""
This fucntion is to pre process the sequences with the notes file from the above function

"""
def prepare_sequences(notes, n_vocab):
    pitchnames = sorted(set(item for item in notes))
    note_to_int = dict((notes, number) for number, notes in enumerate(pitchnames))
    print("Dictionary size: %f" % len(note_to_int))

    sequence_length = 100

    network_input = []
    network_output = []

  
    print("Create input sequences and the corresponding outputs")
    
    """	
    For each input sequence, we create a corresponding output sequence
"""
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])


    n_patterns = len(network_input)
	
	
    """
	Re shaping the input to fit into the LSTM compatible format
    """
    network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    
    """
		Normalising the input

"""
    network_input = network_input / float(n_vocab)

    network_output = np_utils.to_categorical(network_output)
    return (network_input, network_output)
"""
This funtion generates the network
"""

def create_network(network_input, n_vocab):
  
  
    print("Creating model")
    """
	Creating a sequential model
	
"""
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        return_sequences=True
    ))
	
    """
	We add the various factors sucha s the sensity, network input and output
"""
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model

"""
Once the model is created, we train the model and generate weights which will be used to generate music.

The hdf5 type of files can be easili procesed to generate MIDI files using Music21
"""
def train(model, network_input, network_output):
    """
	We train the models and generate a set of weights dependiing on the number of cycles. In this case 200.
	"""
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
	
    """
	This is the checkpoit where we monito the loass in each cycle of the training
"""
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]
	
    """
	Model Fitting
"""
    model.fit(network_input, network_output, epochs=200, batch_size=64, callbacks=callbacks_list)


if __name__ == '__main__':
    notes = get_notes()

    n_vocab = len(set(notes))

    network_input, network_output = prepare_sequences(notes, n_vocab)

    model = create_network(network_input, n_vocab)

    train(model, network_input, network_output)
