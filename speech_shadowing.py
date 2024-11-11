import soundfile as sf
import numpy as np
import librosa
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from pathlib import Path
from pydub import AudioSegment
import librosa.display
import matplotlib.pyplot as plt
import parselmouth  # For detailed pitch analysis
from parselmouth.praat import call  # For formant analysis
import scipy.stats
from typing import Dict, List, Tuple

class ShadowingPractice:
    def __init__(self):
        self.output_dir = Path("recordings")
        self.output_dir.mkdir(exist_ok=True)
        self.pronunciation_weights = {
            'formants': 0.4,
            'spectral': 0.3,
            'mfcc': 0.3
        }
        
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
            'tempo': librosa.feature.tempo(y=audio, sr=sr)[0],
            'rhythm': librosa.feature.tempogram(y=audio, sr=sr),
            'chroma': librosa.feature.chroma_stft(y=audio, sr=sr),
            'formants': self._extract_formants(audio, sr),
            'zero_crossing_rate': librosa.feature.zero_crossing_rate(audio)[0]
        }
        return features

    def _extract_formants(self, audio, sr):
        """Extract formants using Praat"""
        # Ensure audio is a 1D array
        if audio.ndim > 1:
            audio = audio.flatten()
        
        # Replace NaN values with zeros and convert to float64
        audio = np.nan_to_num(audio).astype(np.float64)
        
        # Convert audio to Praat Sound object
        sound = parselmouth.Sound(audio, sampling_frequency=float(sr))
        formant = call(sound, "To Formant (burg)", 0.0025, 5, 5500, 0.025, 50)
        
        # Extract first 3 formants
        formants = []
        for i in range(1, 4):
            formant_values = [call(formant, "Get value at time", i, t, 'Hertz', 'Linear') 
                            for t in np.arange(0, sound.duration, 0.01)]
            formants.append(np.array(formant_values))
        
        return np.array(formants)

    def analyze_detailed_differences(self, target_features, recorded_features, dtw_path, sr):
        """Provide detailed analysis of differences"""
        feedback = {
            'pronunciation': self.analyze_pronunciation(target_features, recorded_features, sr),
            'rhythm': self.analyze_rhythm(target_features, recorded_features, sr),
            'sound_strength': self.analyze_sound_strength(target_features, recorded_features, sr),
            'intonation': self.analyze_intonation(target_features, recorded_features, sr)
        }
        
        return feedback

    def analyze_pronunciation(self, target_features, recorded_features, sr) -> Dict:
        """Detailed pronunciation analysis"""
        feedback = {
            'score': 0,
            'details': [],
            'segments': []
        }
        
        # Analyze formants
        formant_score, formant_feedback = self._compare_formants(
            target_features['formants'],
            recorded_features['formants']
        )
        
        # Analyze spectral characteristics
        spectral_score, spectral_feedback = self._compare_spectral(
            target_features['spectral_centroid'],
            recorded_features['spectral_centroid'],
            target_features['zero_crossing_rate'],
            recorded_features['zero_crossing_rate']
        )
        
        # Analyze MFCC for overall pronunciation
        mfcc_score, mfcc_feedback = self._compare_mfcc(
            target_features['mfcc'],
            recorded_features['mfcc']
        )
        
        # Calculate weighted score
        feedback['score'] = (
            formant_score * self.pronunciation_weights['formants'] +
            spectral_score * self.pronunciation_weights['spectral'] +
            mfcc_score * self.pronunciation_weights['mfcc']
        )
        
        feedback['details'].extend(formant_feedback + spectral_feedback + mfcc_feedback)
        
        return feedback

    def analyze_rhythm(self, target_features, recorded_features, sr) -> Dict:
        """Detailed rhythm analysis"""
        feedback = {
            'score': 0,
            'details': [],
            'segments': []
        }
        
        # Compare onset patterns
        target_onsets = librosa.frames_to_time(target_features['onset_frames'], sr=sr)
        recorded_onsets = librosa.frames_to_time(recorded_features['onset_frames'], sr=sr)
        
        # Calculate inter-onset intervals
        target_ioi = np.diff(target_onsets)
        recorded_ioi = np.diff(recorded_onsets)
        
        # Compare rhythm patterns using tempogram
        # Ensure both tempograms have the same size
        target_rhythm = target_features['rhythm']
        recorded_rhythm = recorded_features['rhythm']
        
        # Get minimum size along each dimension
        min_rows = min(target_rhythm.shape[0], recorded_rhythm.shape[0])
        min_cols = min(target_rhythm.shape[1], recorded_rhythm.shape[1])
        
        # Truncate both arrays to the same size
        target_rhythm = target_rhythm[:min_rows, :min_cols]
        recorded_rhythm = recorded_rhythm[:min_rows, :min_cols]
        
        # Calculate correlation
        rhythm_correlation = np.corrcoef(
            target_rhythm.flatten(),
            recorded_rhythm.flatten()
        )[0, 1]
        
        # Calculate rhythm score
        rhythm_score = (rhythm_correlation + 1) / 2 * 100
        feedback['score'] = rhythm_score
        
        # Generate detailed feedback
        if rhythm_score < 70:
            feedback['details'].append("Significant rhythm differences detected")
            
            # Analyze specific rhythm patterns
            if len(target_ioi) > len(recorded_ioi):
                feedback['details'].append("Missing syllables or words detected")
            elif len(target_ioi) < len(recorded_ioi):
                feedback['details'].append("Extra syllables or words detected")
                
            # Compare rhythm stability
            target_stability = np.std(target_ioi)
            recorded_stability = np.std(recorded_ioi)
            if recorded_stability > target_stability * 1.2:
                feedback['details'].append("Rhythm is unstable - try to maintain more consistent timing")
        
        return feedback

    def analyze_sound_strength(self, target_features, recorded_features, sr) -> Dict:
        """Detailed sound strength/volume analysis"""
        feedback = {
            'score': 0,
            'details': [],
            'segments': []
        }
        
        # Normalize energy features
        target_energy = target_features['energy'] / np.max(target_features['energy'])
        recorded_energy = recorded_features['energy'] / np.max(recorded_features['energy'])
        
        # Ensure both arrays have the same length
        min_length = min(len(target_energy), len(recorded_energy))
        target_energy = target_energy[:min_length]
        recorded_energy = recorded_energy[:min_length]
        
        # Calculate dynamic range
        target_dynamic_range = np.ptp(target_energy)
        recorded_dynamic_range = np.ptp(recorded_energy)
        
        # Compare energy patterns
        energy_correlation = np.corrcoef(target_energy, recorded_energy)[0, 1]
        energy_score = (energy_correlation + 1) / 2 * 100
        
        feedback['score'] = energy_score
        
        # Analyze dynamic range differences
        if abs(target_dynamic_range - recorded_dynamic_range) > 0.2:
            if recorded_dynamic_range < target_dynamic_range:
                feedback['details'].append(
                    "Dynamic range is too narrow - try to vary your volume more between loud and soft parts"
                )
            else:
                feedback['details'].append(
                    "Dynamic range is too wide - try to control volume variations"
                )
        
        # Find specific segments with volume mismatches
        segments = self._find_volume_segments(target_energy, recorded_energy, sr)
        feedback['segments'] = segments
        
        return feedback

    def analyze_intonation(self, target_features, recorded_features, sr) -> Dict:
        """Detailed intonation analysis"""
        feedback = {
            'score': 0,
            'details': [],
            'segments': []
        }
        
        # Extract pitch contours and ensure they're the same length
        target_pitch = target_features['pitch']
        recorded_pitch = recorded_features['pitch']
        
        # Get minimum length and truncate both arrays
        min_length = min(len(target_pitch), len(recorded_pitch))
        target_pitch = target_pitch[:min_length]
        recorded_pitch = recorded_pitch[:min_length]
        
        # Replace any NaN values with 0
        target_pitch = np.nan_to_num(target_pitch)
        recorded_pitch = np.nan_to_num(recorded_pitch)
        
        # Calculate pitch statistics
        target_stats = self._calculate_pitch_statistics(target_pitch)
        recorded_stats = self._calculate_pitch_statistics(recorded_pitch)
        
        # Compare pitch ranges
        pitch_range_diff = abs(target_stats['range'] - recorded_stats['range'])
        if pitch_range_diff > 20:
            feedback['details'].append(
                f"Pitch range differs by {pitch_range_diff:.1f} Hz - "
                f"{'wider' if recorded_stats['range'] > target_stats['range'] else 'narrower'} than target"
            )
        
        # Compare pitch patterns
        pitch_correlation = np.corrcoef(target_pitch, recorded_pitch)[0, 1]
        
        intonation_score = (pitch_correlation + 1) / 2 * 100
        feedback['score'] = intonation_score
        
        # Find specific segments with intonation mismatches
        segments = self._find_intonation_segments(target_pitch, recorded_pitch, sr)
        feedback['segments'] = segments
        
        return feedback

    def _calculate_pitch_statistics(self, pitch):
        """Calculate various statistics for pitch analysis"""
        valid_pitch = pitch[pitch > 0]
        return {
            'mean': np.mean(valid_pitch),
            'std': np.std(valid_pitch),
            'range': np.ptp(valid_pitch),
            'median': np.median(valid_pitch)
        }

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

    def _compare_formants(self, target_formants, recorded_formants):
        """Compare formants between target and recorded audio"""
        # Ensure both arrays have the same length
        min_length = min(len(target_formants[0]), len(recorded_formants[0]))
        
        # Calculate correlation for each formant
        correlations = []
        for i in range(3):  # First 3 formants
            target_f = target_formants[i][:min_length]
            recorded_f = recorded_formants[i][:min_length]
            # Handle edge cases
            if np.all(target_f == target_f[0]) or np.all(recorded_f == recorded_f[0]):
                correlation = 0.0
            else:
                try:
                    correlation = np.corrcoef(target_f, recorded_f)[0, 1]
                    if np.isnan(correlation):
                        correlation = 0.0
                except:
                    correlation = 0.0
            correlations.append(correlation)
        
        # Calculate overall score
        score = max(0, min(100, np.mean(correlations) * 100))
        
        # Generate feedback
        feedback = []
        if score < 70:
            feedback.append("Significant differences in vowel pronunciation detected")
            
        return score, feedback

    def _compare_spectral(self, target_centroid, recorded_centroid, target_zcr, recorded_zcr):
        """Compare spectral characteristics between target and recorded audio"""
        # Ensure arrays have same length
        min_length = min(len(target_centroid), len(recorded_centroid))
        target_centroid = target_centroid[:min_length]
        recorded_centroid = recorded_centroid[:min_length]
        target_zcr = target_zcr[:min_length]
        recorded_zcr = recorded_zcr[:min_length]
        
        # Calculate correlations with error handling
        try:
            centroid_correlation = np.corrcoef(target_centroid, recorded_centroid)[0, 1]
            if np.isnan(centroid_correlation):
                centroid_correlation = 0.0
        except:
            centroid_correlation = 0.0
            
        try:
            zcr_correlation = np.corrcoef(target_zcr, recorded_zcr)[0, 1]
            if np.isnan(zcr_correlation):
                zcr_correlation = 0.0
        except:
            zcr_correlation = 0.0
        
        # Calculate overall spectral score
        score = max(0, min(100, ((centroid_correlation + zcr_correlation) / 2) * 100))
        
        # Generate feedback
        feedback = []
        if score < 70:
            if centroid_correlation < 0.7:
                feedback.append("Significant differences in sound brightness/timbre detected")
            if zcr_correlation < 0.7:
                feedback.append("Differences in consonant pronunciation detected")
                
        return score, feedback

    def _compare_mfcc(self, target_mfcc, recorded_mfcc):
        """Compare MFCC features between target and recorded audio"""
        # Ensure both MFCCs have same length
        min_length = min(target_mfcc.shape[1], recorded_mfcc.shape[1])
        target_mfcc = target_mfcc[:, :min_length]
        recorded_mfcc = recorded_mfcc[:, :min_length]
        
        # Calculate correlation for each MFCC coefficient
        correlations = []
        for i in range(min(target_mfcc.shape[0], recorded_mfcc.shape[0])):
            try:
                correlation = np.corrcoef(target_mfcc[i], recorded_mfcc[i])[0, 1]
                if np.isnan(correlation):
                    correlation = 0.0
            except:
                correlation = 0.0
            correlations.append(correlation)
        
        # Calculate overall score
        score = max(0, min(100, np.mean(correlations) * 100))
        
        # Generate feedback
        feedback = []
        if score < 70:
            feedback.append("Overall pronunciation patterns show significant differences")
            
        return score, feedback

    def _find_volume_segments(self, target_energy, recorded_energy, sr):
        """Find segments with significant volume differences"""
        segments = []
        
        # Calculate the difference in energy
        energy_diff = np.abs(target_energy - recorded_energy)
        
        # Find segments where the difference is significant (threshold can be adjusted)
        threshold = np.mean(energy_diff) + np.std(energy_diff)
        significant_diffs = np.where(energy_diff > threshold)[0]
        
        # Group consecutive frames into segments
        if len(significant_diffs) > 0:
            groups = self.group_consecutive_frames(significant_diffs)
            
            # Convert frame indices to time segments
            hop_length = 512  # This should match the hop_length used in RMS calculation
            for start_frame, end_frame in groups:
                start_time = librosa.frames_to_time(start_frame, sr=sr, hop_length=hop_length)
                end_time = librosa.frames_to_time(end_frame, sr=sr, hop_length=hop_length)
                
                # Add segment description
                avg_diff = np.mean(energy_diff[start_frame:end_frame])
                if target_energy[start_frame] > recorded_energy[start_frame]:
                    direction = "too quiet"
                else:
                    direction = "too loud"
                    
                segments.append(
                    f"Time {start_time:.2f}s - {end_time:.2f}s: {direction}"
                )
        
        return segments

    def _find_intonation_segments(self, target_pitch, recorded_pitch, sr):
        """Find segments with significant intonation differences"""
        segments = []
        
        # Calculate the difference in pitch
        pitch_diff = np.abs(target_pitch - recorded_pitch)
        
        # Find segments where the difference is significant
        threshold = np.mean(pitch_diff) + np.std(pitch_diff)
        significant_diffs = np.where(pitch_diff > threshold)[0]
        
        # Group consecutive frames into segments
        if len(significant_diffs) > 0:
            groups = self.group_consecutive_frames(significant_diffs)
            
            # Convert frame indices to time segments
            hop_length = 512  # This should match the hop_length used in feature extraction
            for start_frame, end_frame in groups:
                start_time = librosa.frames_to_time(start_frame, sr=sr, hop_length=hop_length)
                end_time = librosa.frames_to_time(end_frame, sr=sr, hop_length=hop_length)
                
                # Determine if pitch is too high or too low
                if target_pitch[start_frame] > recorded_pitch[start_frame]:
                    direction = "too low"
                else:
                    direction = "too high"
                    
                segments.append(
                    f"Time {start_time:.2f}s - {end_time:.2f}s: pitch is {direction}"
                )
        
        return segments

def main():
    practice = ShadowingPractice()
    
    # Get target audio file path
    target_file = Path("target_audio.wav")
    if not target_file.exists():
        print(f"Error: Target audio file not found at {target_file}")
        return
    
    # Use a local audio file for the user's shadowing attempt
    # recorded_file = Path("user_audio.wav")
    recorded_file = Path("user_audioaaa.wav")
    if not recorded_file.exists():
        print(f"Error: User audio file not found at {recorded_file}")
        return
    
    # Compare audio and get score
    score, feedback = practice.compare_audio(target_file, recorded_file)
    
    # Display detailed results
    print(f"\nOverall Score: {score:.2f}/100")
    
    if feedback:
        print("\nDetailed Feedback:")
        
        for category, results in feedback.items():
            print(f"\n{category.upper()} Score: {results['score']:.2f}/100")
            
            if results['details']:
                print("Details:")
                for detail in results['details']:
                    print(f"- {detail}")
            
            if results['segments']:
                print("Specific segments to improve:")
                for segment in results['segments']:
                    print(f"- {segment}")
    else:
        print("\nGreat job! Your pronunciation is very close to the target.")
        
    print(f"\nYour recording has been saved to: {recorded_file}")

if __name__ == "__main__":
    main() 