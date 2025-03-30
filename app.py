from flask import Flask, request, jsonify, render_template, send_file
import os
import json
from melody_generator import MelodyGenerator
from preprocess import SEQUENCE_LENGTH
import tensorflow as tf
import numpy as np
import music21 as m21
import io
import base64

app = Flask(__name__, static_folder="static", template_folder="templates")

# Initialize the melody generator
mg = MelodyGenerator()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_melody():
    data = request.json
    seed = data.get('seed', "60 _ 64 _ 67 _")
    num_steps = int(data.get('steps', 500))
    temperature = float(data.get('temperature', 0.3))
    
    try:
        # Generate the melody
        melody = mg.generate_melody(seed, num_steps, SEQUENCE_LENGTH, temperature)
        
        # Save to a MIDI file
        output_file = "static/generated_melody.mid"
        mg.save_melody(melody, file_name=output_file)
        
        return jsonify({
            'success': True,
            'melody': melody,
            'midi_file': output_file
        })
    except Exception as e:
        print(f"Error generating melody: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/play_seed', methods=['POST'])
def play_seed():
    data = request.json
    seed = data.get('seed', "60 _ 64 _ 67 _")
    
    try:
        # Process and save the seed as MIDI
        seed_melody = seed.split()
        output_file = "static/seed_melody.mid"
        mg.save_melody(seed_melody, file_name=output_file)
        
        return jsonify({
            'success': True,
            'midi_file': output_file
        })
    except Exception as e:
        print(f"Error playing seed: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    # Create static directory if it doesn't exist
    os.makedirs('static', exist_ok=True)
    app.run(debug=True)