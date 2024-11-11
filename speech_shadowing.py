import soundfile as sf
import numpy as np
import librosa
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from pathlib import Path
from pydub import AudioSegment
import librosa.display
import matplotlib.pyplot as plt

class ShadowingPractice:
    def __init__(self):
        self.output_dir = Path("recordings")
        self.output_dir.mkdir(exist_ok=True)
        
    def convert_to_wav(self, audio_path):
        """Convert audio file to WAV format if needed"""
        audio_path = Path(audio_path)
        if audio_path.suffix.lower() in ['.wav']:
            return audio_path
            
        # Convert to WAV
        wav_path = self.output_dir / f"{audio_path.stem}.wav"
        audio = AudioSegment.from_file(str(audio_path))
        audio.export(str(wav_path), format='wav')
        return wav_path
        
    def compare_audio(self, target_path, recorded_path):
        """Compare target and recorded audio using DTW"""
        print(f"Comparing {target_path} and {recorded_path}")
        
        # Convert audio files to WAV if needed
        target_wav = self.convert_to_wav(target_path)
        recorded_wav = self.convert_to_wav(recorded_path)
        
        # Load both audio files
        target_audio, sr = librosa.load(str(target_wav))
        recorded_audio, _ = librosa.load(str(recorded_wav), sr=sr)
        
        # Extract various features for detailed analysis
        target_features = self.extract_audio_features(target_audio, sr)
        recorded_features = self.extract_audio_features(recorded_audio, sr)
        
        # Calculate overall score using MFCC
        distance, path = fastdtw(target_features['mfcc'].T, recorded_features['mfcc'].T, dist=euclidean)
        avg_distance = distance / max(target_features['mfcc'].shape[1], recorded_features['mfcc'].shape[1])
        score = 100 * np.exp(-avg_distance / 100)
        score = np.clip(score, 0, 100)
        
        # Get detailed feedback
        feedback = self.analyze_detailed_differences(
            target_features, 
            recorded_features,
            path,
            sr
        )
        
        return score, feedback

    def extract_audio_features(self, audio, sr):
        """Extract multiple audio features for detailed analysis"""
        features = {
            'mfcc': librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, hop_length=512),
            'pitch': librosa.yin(audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')),
            'energy': librosa.feature.rms(y=audio, hop_length=512)[0],
            'spectral_centroid': librosa.feature.spectral_centroid(y=audio, sr=sr)[0],
            'onset_frames': librosa.onset.onset_detect(y=audio, sr=sr, units='frames'),
            'tempo': librosa.beat.tempo(y=audio, sr=sr)[0]
        }
        return features

    def analyze_detailed_differences(self, target_features, recorded_features, dtw_path, sr):
        """Provide detailed analysis of differences"""
        feedback = []
        
        # Analyze timing
        target_tempo = target_features['tempo']
        recorded_tempo = recorded_features['tempo']
        tempo_diff = abs(target_tempo - recorded_tempo)
        if tempo_diff > 10:
            feedback.append(f"Speaking speed differs by {tempo_diff:.1f} BPM "
                          f"({'faster' if recorded_tempo > target_tempo else 'slower'} than target)")

        # Analyze pitch patterns
        target_pitch = target_features['pitch']
        recorded_pitch = recorded_features['pitch']
        
        # Find segments with significant pitch differences
        pitch_segments = self.find_pitch_differences(target_pitch, recorded_pitch, sr)
        if pitch_segments:
            feedback.extend(pitch_segments)

        # Analyze energy/volume patterns
        target_energy = target_features['energy']
        recorded_energy = recorded_features['energy']
        energy_segments = self.find_energy_differences(target_energy, recorded_energy, sr)
        if energy_segments:
            feedback.extend(energy_segments)

        # Analyze pronunciation clarity
        spectral_diff = self.analyze_spectral_differences(
            target_features['spectral_centroid'],
            recorded_features['spectral_centroid'],
            sr
        )
        if spectral_diff:
            feedback.extend(spectral_diff)

        return feedback

    def find_pitch_differences(self, target_pitch, recorded_pitch, sr):
        """Identify segments with significant pitch differences"""
        feedback = []
        min_segment_length = int(0.2 * sr)  # 200ms minimum segment
        
        # Normalize and compare pitch contours
        target_pitch = np.nan_to_num(target_pitch)
        recorded_pitch = np.nan_to_num(recorded_pitch)
        
        # Interpolate to make both arrays the same length
        target_len = len(target_pitch)
        recorded_len = len(recorded_pitch)
        max_len = max(target_len, recorded_len)
        
        # Create evenly spaced points for interpolation
        x_target = np.linspace(0, 1, target_len)
        x_recorded = np.linspace(0, 1, recorded_len)
        x_new = np.linspace(0, 1, max_len)
        
        # Interpolate both signals to the same length
        target_interpolated = np.interp(x_new, x_target, target_pitch)
        recorded_interpolated = np.interp(x_new, x_recorded, recorded_pitch)
        
        # Find segments where pitch differs significantly
        diff_threshold = 50  # Hz
        significant_diffs = np.where(abs(target_interpolated - recorded_interpolated) > diff_threshold)[0]
        
        if len(significant_diffs) > 0:
            segments = self.group_consecutive_frames(significant_diffs)
            for start, end in segments:
                if (end - start) >= min_segment_length:
                    # Convert frame index to time
                    time_start = start * len(target_pitch) / max_len / sr
                    feedback.append(f"Pitch difference detected at {time_start:.2f}s - "
                                  f"Try matching the target intonation pattern")
        
        return feedback

    def find_energy_differences(self, target_energy, recorded_energy, sr):
        """Identify segments with significant energy/volume differences"""
        feedback = []
        
        # Normalize energies
        target_energy = target_energy / np.max(target_energy)
        recorded_energy = recorded_energy / np.max(recorded_energy)
        
        # Interpolate to make both arrays the same length
        target_len = len(target_energy)
        recorded_len = len(recorded_energy)
        max_len = max(target_len, recorded_len)
        
        # Create evenly spaced points for interpolation
        x_target = np.linspace(0, 1, target_len)
        x_recorded = np.linspace(0, 1, recorded_len)
        x_new = np.linspace(0, 1, max_len)
        
        # Interpolate both signals to the same length
        target_interpolated = np.interp(x_new, x_target, target_energy)
        recorded_interpolated = np.interp(x_new, x_recorded, recorded_energy)
        
        # Find segments where energy differs significantly
        diff_threshold = 0.3
        significant_diffs = np.where(abs(target_interpolated - recorded_interpolated) > diff_threshold)[0]
        
        if len(significant_diffs) > 0:
            segments = self.group_consecutive_frames(significant_diffs)
            for start, end in segments:
                time_start = librosa.frames_to_time(start, sr=sr, hop_length=512)
                feedback.append(f"Volume mismatch at {time_start:.2f}s - "
                              f"{'Speak louder' if recorded_interpolated[start] < target_interpolated[start] else 'Speak softer'}")
        
        return feedback

    def analyze_spectral_differences(self, target_centroid, recorded_centroid, sr):
        """Analyze differences in pronunciation clarity using spectral centroid"""
        feedback = []
        
        # Compare average spectral centroids
        target_mean = np.mean(target_centroid)
        recorded_mean = np.mean(recorded_centroid)
        
        diff_ratio = abs(target_mean - recorded_mean) / target_mean
        if diff_ratio > 0.2:
            if recorded_mean < target_mean:
                feedback.append("Overall pronunciation could be clearer - "
                              "try to articulate sounds more distinctly")
            else:
                feedback.append("Pronunciation is over-emphasized - "
                              "try to speak more naturally")
        
        return feedback

    @staticmethod
    def group_consecutive_frames(frames):
        """Helper function to group consecutive frame indices"""
        if len(frames) == 0:
            return []
            
        groups = []
        start = frames[0]
        prev = frames[0]
        
        for frame in frames[1:]:
            if frame - prev > 1:
                groups.append((start, prev))
                start = frame
            prev = frame
            
        groups.append((start, prev))
        return groups

def main():
    practice = ShadowingPractice()
    
    # Get target audio file path
    target_file = Path("target_audio.wav")
    if not target_file.exists():
        print(f"Error: Target audio file not found at {target_file}")
        return
    
    # Use a local audio file for the user's shadowing attempt
    # recorded_file = Path("user_audio.wav")
    recorded_file = Path("user_audio.wav")
    if not recorded_file.exists():
        print(f"Error: User audio file not found at {recorded_file}")
        return
    
    # Compare audio and get score
    score, feedback = practice.compare_audio(target_file, recorded_file)
    
    # Display results
    print(f"\nOverall Score: {score:.2f}/100")
    
    if feedback:
        print("\nAreas for Improvement:")
        for area in feedback:
            print(f"- {area}")
    else:
        print("\nGreat job! Your pronunciation is very close to the target.")
        
    print(f"\nYour recording has been saved to: {recorded_file}")

if __name__ == "__main__":
    main() 