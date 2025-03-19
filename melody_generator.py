import json
import numpy as np
import tensorflow.keras as keras
import music21 as m21
from preprocess import SEQUENCE_LENGTH, MAPPING_PATH

class MelodyGenerator:
    """A class that wraps the LSTM model and offers utilities to generate melodies."""

    def __init__(self, model_path="model.h5"):
        """Constructor that initialises TensorFlow model"""

        self.model_path = model_path
        self.model = keras.models.load_model(model_path)

        with open(MAPPING_PATH, "r") as fp:
            self._mappings = json.load(fp)

        self._start_symbols = ["/"] * SEQUENCE_LENGTH


    def generate_melody(self, seed, num_steps, max_sequence_length, temperature):
        """Generates a melody using the DL model and returns a midi file.

        :param seed (str): Melody seed with the notation used to encode the dataset
        :param num_steps (int): Number of steps to be generated
        :param max_sequence_len (int): Max number of steps in seed to be considered for generation
        :param temperature (float): Float in interval [0, 1]. Numbers closer to 0 make the model more deterministic.
            A number closer to 1 makes the generation more unpredictable.

        :return melody (list of str): List with symbols representing a melody
        """

        # create seed with start symbols
        seed = seed.split()
        melody = seed
        seed = self._start_symbols + seed

        # map seed to int
        seed = [self._mappings[symbol] for symbol in seed]

        for _ in range(num_steps):

            # limit the seed to max_sequence_length
            seed = seed[-max_sequence_length:]

            # Make prediction - modify this part
            seed_input = np.array(seed)  # Convert to numpy array
            # Reshape to match expected input shape (batch_size, sequence_length)
            seed_input = seed_input.reshape(1, -1)
            
            # Make prediction
            probabilities = self.model.predict(seed_input)[0]
            
            # Sample with temperature
            output_int = self._sample_with_temperature(probabilities, temperature)

            # update seed
            seed.append(output_int)

            # map int to our encoding
            output_symbol = [k for k, v in self._mappings.items() if v == output_int][0]

            # check whether we're at the end of a melody
            if output_symbol == "/":
                break

            # update melody
            melody.append(output_symbol)

        return melody


    def _sample_with_temperature(self, probabilites, temperature):
        """Samples an index from a probability array reapplying softmax using temperature

        :param predictions (nd.array): Array containing probabilities for each of the possible outputs.
        :param temperature (float): Float in interval [0, 1]. Numbers closer to 0 make the model more deterministic.
            A number closer to 1 makes the generation more unpredictable.

        :return index (int): Selected output symbol
        """
        predictions = np.log(probabilites) / temperature
        probabilites = np.exp(predictions) / np.sum(np.exp(predictions))

        choices = range(len(probabilites)) # [0, 1, 2, 3]
        index = np.random.choice(choices, p=probabilites)

        return index


    def save_melody(self, melody, step_duration=0.25, format="midi", file_name="mel.mid"):
        """Converts a melody into a MIDI file

        :param melody (list of str):
        :param min_duration (float): Duration of each time step in quarter length
        :param file_name (str): Name of midi file
        :return:
        """

        # create a music21 stream
        stream = m21.stream.Stream()

        start_symbol = None
        step_counter = 1

        # parse all the symbols in the melody and create note/rest objects
        for i, symbol in enumerate(melody):

            # handle case in which we have a note/rest
            if symbol != "_" or i + 1 == len(melody):

                # ensure we're dealing with note/rest beyond the first one
                if start_symbol is not None:

                    quarter_length_duration = step_duration * step_counter # 0.25 * 4 = 1

                    # handle rest
                    if start_symbol == "r":
                        m21_event = m21.note.Rest(quarterLength=quarter_length_duration)

                    # handle note
                    else:
                        m21_event = m21.note.Note(int(start_symbol), quarterLength=quarter_length_duration)

                    stream.append(m21_event)

                    # reset the step counter
                    step_counter = 1

                start_symbol = symbol

            # handle case in which we have a prolongation sign "_"
            else:
                step_counter += 1

        # write the m21 stream to a midi file
        stream.write(format, file_name)


if __name__ == "__main__":
    mg = MelodyGenerator()
    
    # Original seeds
    seed1 = "67 _ 67 _ 67 _ _ 65 64 _ 64 _ 64 _ _"  # Original seed
    seed2 = "67 _ _ _ _ _ 65 _ 64 _ 62 _ 60 _ _ _"  # Original seed2
    
    # New alternative seeds
    
    # Simple ascending scale pattern
    scale_seed = "60 _ 62 _ 64 _ 65 _ 67 _ 69 _ 71 _ 72 _"
    
    # Descending pattern
    descending_seed = "72 _ 71 _ 69 _ 67 _ 65 _ 64 _ 62 _ 60 _"
    
    # Arpeggio-like pattern (C major chord)
    arpeggio_seed = "60 _ 64 _ 67 _ 72 _ 67 _ 64 _ 60 _ _ _"
    
    # Rhythmic pattern with rests
    rhythmic_seed = "60 _ _ r 62 _ r 64 _ _ r 65 _ _ _ _"
    
    # Jump pattern with wider intervals
    jump_seed = "60 _ 67 _ 64 _ 72 _ 67 _ 60 _ _ _ _ _"
    
    # Choose which seed to use
    current_seed = arpeggio_seed
    
    # Save the seed as MIDI for comparison
    seed_melody = current_seed.split()
    mg.save_melody(seed_melody, file_name="seed_melody.mid")
    
    # Generate and save the new melody
    melody = mg.generate_melody(current_seed, 500, SEQUENCE_LENGTH, 0.3)
    print(melody)
    mg.save_melody(melody, file_name="generated_melody.mid")
