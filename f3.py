

import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel
import numpy as np

class EmotionExtractionModule(nn.Module):
    """Module for extracting emotional information from literary text"""
    def __init__(self, model_name="roberta-base", num_emotions=8, device=None):
        super(EmotionExtractionModule, self).__init__()

        # Load pre-trained language model for emotion analysis
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(model_name)

        # Emotion classifier head
        self.emotion_classifier = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_emotions),
            nn.Sigmoid()  # Multiple emotions can be present simultaneously
        )

        # Define the emotion categories
        self.emotion_categories = [
            "joy", "sadness", "anger", "fear",
            "tenderness", "excitement", "calmness", "tension"
        ]

        # Sliding window parameters for analyzing longer texts
        self.window_size = 512  # Max tokens per window
        self.stride = 256  # Overlap between windows

        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

    def forward(self, text, return_attention=False):
        """
        Extract emotion scores from text
        Returns emotion scores and attention weights for visualization
        """
        # For longer texts, break into overlapping chunks
        if isinstance(text, str):
            text = [text]

        all_emotion_scores = []
        all_attention_weights = []

        for t in text:
            # Break long text into chunks with sliding window
            chunks = self._create_text_chunks(t)
            chunk_emotions = []
            chunk_attentions = []

            for chunk in chunks:
                # Tokenize with attention mask
                inputs = self.tokenizer(chunk, return_tensors="pt",
                                      padding=True, truncation=True, max_length=self.window_size)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}  # Fixed to self.device

                # Get model outputs
                with torch.no_grad():
                    outputs = self.model(**inputs, output_attentions=return_attention)  # Only get attention if needed

                # Use [CLS] token representation for classification
                sequence_output = outputs.last_hidden_state[:, 0, :]
                emotions = self.emotion_classifier(sequence_output)

                # Store results
                chunk_emotions.append(emotions)

                # Extract attention patterns only if requested
                if return_attention:
                    attentions = outputs.attentions[-1].mean(dim=1)  # Average over heads in last layer
                    chunk_attentions.append(attentions)

            # Combine emotions from chunks with temporal weighting
            if len(chunk_emotions) > 1:
                chunk_emotions = torch.cat(chunk_emotions, dim=0)
                # Exponential weighting might work better for many chunks
                weights = torch.exp(torch.linspace(0.0, 1.0, len(chunk_emotions))).to(chunk_emotions.device)
                weights = weights / weights.sum()  # Normalize
                weighted_emotions = chunk_emotions * weights.unsqueeze(1)
                final_emotions = weighted_emotions.sum(dim=0, keepdim=True)
            else:
                final_emotions = chunk_emotions[0]

            all_emotion_scores.append(final_emotions)
            if return_attention:
                all_attention_weights.append(chunk_attentions)

        # Combine results from batch
        emotion_scores = torch.cat(all_emotion_scores, dim=0)

        result = {
            "emotion_scores": emotion_scores,
            "emotion_categories": self.emotion_categories
        }

        if return_attention:
            result["attention_weights"] = all_attention_weights

        return result

    def _create_text_chunks(self, text):
        """Create overlapping chunks for long text analysis"""
        # Quick tokenize to get token count (without padding/truncation)
        tokens = self.tokenizer.encode(text)

        if len(tokens) <= self.window_size:
            return [text]  # Text fits in one window

        # For longer texts, create overlapping chunks
        chunks = []
        words = text.split()

        # Estimate words per window based on tokens
        words_per_token = max(1, len(words) / len(tokens))
        words_per_window = int(self.window_size * words_per_token * 0.9)  # 90% to be safe
        stride_in_words = int(self.stride * words_per_token * 0.9)

        # Create overlapping chunks
        for i in range(0, len(words), stride_in_words):
            chunk = " ".join(words[i:i + words_per_window])
            chunks.append(chunk)

            # Stop if we've covered the whole text
            if i + words_per_window >= len(words):
                break

        return chunks

    def extract_emotional_arc(self, text, num_segments=10):
        """
        Extract emotional arc across text for narrative progression
        Returns emotion scores for multiple segments of text
        """
        # Create segments
        if isinstance(text, str):
            words = text.split()
            segment_size = max(1, len(words) // num_segments)
            segments = []

            for i in range(0, len(words), segment_size):
                segment = " ".join(words[i:i + segment_size])
                segments.append(segment)

                # Stop if we've covered the whole text
                if len(segments) >= num_segments:
                    break
        else:
            segments = text  # Already a list of segments

        # Process each segment
        segment_emotions = []

        for segment in segments:
            result = self.forward([segment], return_attention=return_attention)
            segment_emotions.append(result["emotion_scores"][0])

        # Stack emotions to create emotional arc [num_segments, num_emotions]
        emotional_arc = torch.stack(segment_emotions)

        return {
            "emotional_arc": emotional_arc,
            "emotion_categories": self.emotion_categories,
            "segments": segments
        }

# Fix for the EmotionToMusicMapper class
class EmotionToMusicMapper(nn.Module):
    """Maps emotional content to musical parameters"""
    def __init__(self, emotion_dim=8, music_param_dim=16, device=None):
        super(EmotionToMusicMapper, self).__init__()
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.emotion_dim = emotion_dim
        self.music_param_dim = music_param_dim

        # Neural mapping from emotions to musical parameters
        self.mapping_network = nn.Sequential(
            nn.Linear(emotion_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, music_param_dim)
        )
        self.mapping_network = self.mapping_network.to(self.device)

        # Define musical parameter ranges
        self.music_params = {
            "tempo": {"min": 60, "max": 180},        # BPM
            "key": {"min": 0, "max": 11},            # C=0, C#=1, etc.
            "mode": {"min": 0, "max": 1},            # Major=0, Minor=1
            "rhythm_density": {"min": 0.2, "max": 0.9},
            "note_duration": {"min": 0.1, "max": 0.8},
            "articulation": {"min": 0.2, "max": 0.9}, # Staccato to legato
            "dynamics": {"min": 0.1, "max": 0.9},     # Soft to loud
            "timbre_brightness": {"min": 0.1, "max": 0.9},
            "harmonic_complexity": {"min": 0.1, "max": 0.9},
            "dissonance": {"min": 0.0, "max": 0.7},
            "reverb": {"min": 0.0, "max": 0.8},
            "stereo_width": {"min": 0.3, "max": 1.0},
            "instrumentation": {"min": 0, "max": 5},  # Different ensemble types
            "melodic_range": {"min": 12, "max": 36},  # Range in semitones
            "bass_presence": {"min": 0.1, "max": 0.9},
            "density": {"min": 0.1, "max": 0.9}       # Sparse to dense
        }

        # Emotion to music parameter mapping rules (prior knowledge)
        # These serve as biases for the neural mapping
        self.emotion_music_rules = {
            "joy": {
                "tempo": 0.7,          # Faster tempo
                "mode": 0.2,           # Tends toward major
                "dissonance": 0.2,     # Low dissonance
                "dynamics": 0.7        # Louder dynamics
            },
            "sadness": {
                "tempo": 0.3,          # Slower tempo
                "mode": 0.8,           # Tends toward minor
                "note_duration": 0.7,  # Longer notes
                "reverb": 0.7          # More reverb
            },
            "anger": {
                "tempo": 0.8,          # Fast tempo
                "dissonance": 0.7,     # High dissonance
                "dynamics": 0.9,       # Loud dynamics
                "articulation": 0.3    # More staccato
            },
            "fear": {
                "dissonance": 0.6,     # Higher dissonance
                "dynamics": 0.4,       # Varied dynamics
                "stereo_width": 0.8    # Wide stereo field
            },
            "tenderness": {
                "tempo": 0.4,          # Moderate-slow tempo
                "dynamics": 0.3,       # Soft dynamics
                "articulation": 0.8,   # Legato
                "harmonic_complexity": 0.4  # Simple harmonies
            },
            "excitement": {
                "tempo": 0.8,          # Fast tempo
                "rhythm_density": 0.8, # Dense rhythms
                "dynamics": 0.8        # Loud dynamics
            },
            "calmness": {
                "tempo": 0.3,          # Slow tempo
                "dynamics": 0.3,       # Soft dynamics
                "reverb": 0.6,         # More reverb
                "dissonance": 0.1      # Consonant harmonies
            },
            "tension": {
                "harmonic_complexity": 0.7,  # Complex harmonies
                "dissonance": 0.6,     # More dissonance
                "dynamics": 0.5        # Varied dynamics
            }
        }

    def forward(self, emotion_scores):
        """
        Map emotion scores to musical parameters
        emotion_scores: [batch_size, emotion_dim] or [emotion_dim]
        """
        # Ensure emotion_scores has batch dimension
        if len(emotion_scores.shape) == 1:
            emotion_scores = emotion_scores.unsqueeze(0)

        # Make sure it's on the right device
        emotion_scores = emotion_scores.to(self.device)

        # Apply neural mapping
        raw_params = self.mapping_network(emotion_scores)

        # Apply prior knowledge as bias
        music_params = self._apply_emotion_rules(emotion_scores, raw_params)

        # Scale parameters to their defined ranges
        scaled_params = self._scale_to_ranges(music_params)

        return {
            "music_params": music_params,
            "scaled_params": scaled_params
        }

    def _apply_emotion_rules(self, emotion_scores, raw_params):
        """Apply emotion-music rules as biases to neural output"""
        batch_size = emotion_scores.shape[0]
        device = emotion_scores.device

        # Initialize parameter tensor with neural output
        music_params = raw_params.clone()

        # Get the indices for each parameter in the output tensor
        param_indices = {param: i for i, param in enumerate(self.music_params.keys())}

        # Map emotion indices to names
        emotion_names = ["joy", "sadness", "anger", "fear",
                         "tenderness", "excitement", "calmness", "tension"]

        # For each emotion, apply its rules based on strength
        for b in range(batch_size):
            for i, emotion in enumerate(emotion_names):
                if i >= emotion_scores.shape[1]:
                    continue  # Skip if index is out of bounds

                # Get emotion strength (0 to 1)
                emotion_strength = emotion_scores[b, i].item()

                # Skip if emotion is not strongly present
                if emotion_strength < 0.2:
                    continue

                # Apply each rule for this emotion
                if emotion in self.emotion_music_rules:
                    for param, value in self.emotion_music_rules[emotion].items():
                        if param in param_indices:
                            idx = param_indices[param]
                            # Blend neural output with rule-based value based on emotion strength
                            blend_factor = emotion_strength * 0.7  # Max 70% influence
                            music_params[b, idx] = (1 - blend_factor) * music_params[b, idx] + blend_factor * value

        return music_params

    def _scale_to_ranges(self, music_params):
        """Scale normalized parameters to their actual ranges"""
        batch_size = music_params.shape[0]
        scaled_params = {}

        for i, (param, range_info) in enumerate(self.music_params.items()):
            # Get parameter values (clamped to 0-1)
            values = torch.clamp(music_params[:, i], 0.0, 1.0)

            # Scale to actual range
            min_val, max_val = range_info["min"], range_info["max"]
            scaled = min_val + values * (max_val - min_val)

            # Special handling for discrete parameters
            if param in ["key", "instrumentation"]:
                scaled = scaled.round()

            scaled_params[param] = scaled

        return scaled_params

    def generate_musiclm_prompt(self, emotion_scores, emotion_categories):
        """
        Generate a detailed MusicLM-compatible prompt based on emotion analysis
        emotion_scores: tensor of emotion scores
        emotion_categories: list of emotion category names
        """
        # Ensure emotion_scores has batch dimension and is on CPU for processing
        if len(emotion_scores.shape) == 1:
            emotion_scores = emotion_scores.unsqueeze(0)

        emotion_scores = emotion_scores.detach().cpu()

        # Get music parameters for these emotions
        music_params = self.forward(emotion_scores)
        scaled_params = music_params["scaled_params"]

        # Build prompt
        prompts = []

        for b in range(emotion_scores.shape[0]):
            # Get top emotions
            if emotion_scores.shape[1] <= len(emotion_categories):
                emotions_data = [(emotion_categories[i], emotion_scores[b, i].item())
                                for i in range(emotion_scores.shape[1])]
            else:
                emotions_data = [(f"Emotion {i}", emotion_scores[b, i].item())
                                for i in range(emotion_scores.shape[1])]

            # Sort emotions by strength
            emotions_data.sort(key=lambda x: x[1], reverse=True)

            # Get top 3 emotions
            top_emotions = [e for e, s in emotions_data if s > 0.2][:3]

            # Get key musical parameters
            tempo = scaled_params["tempo"][b].item()

            # Determine mode (major/minor)
            mode = "minor" if scaled_params["mode"][b].item() > 0.5 else "major"

            # Determine instrumentation based on emotions
            instruments = self._choose_instrumentation(top_emotions)

            # Determine musical style based on emotions and parameters
            style = self._choose_style(top_emotions, scaled_params, b)

            # Build descriptive prompt
            emotion_desc = " and ".join(top_emotions) if top_emotions else "neutral"

            prompt = f"A {style} piece in {mode} key at {tempo:.0f} BPM, evoking feelings of {emotion_desc}. "
            prompt += f"Featuring {instruments}. "

            # Add specifics based on parameters
            if scaled_params["harmonic_complexity"][b].item() > 0.7:
                prompt += "Complex harmonies with unexpected chord changes. "
            elif scaled_params["harmonic_complexity"][b].item() < 0.3:
                prompt += "Simple, consonant harmonies. "

            if scaled_params["rhythm_density"][b].item() > 0.7:
                prompt += "Dense, intricate rhythms. "
            elif scaled_params["rhythm_density"][b].item() < 0.3:
                prompt += "Sparse, spacious rhythms. "

            if scaled_params["dynamics"][b].item() > 0.7:
                prompt += "Dramatic dynamic range with powerful crescendos. "
            elif scaled_params["dynamics"][b].item() < 0.3:
                prompt += "Gentle, subtle dynamics. "

            if scaled_params["reverb"][b].item() > 0.6:
                prompt += "Immersive, spacious reverb. "

            prompts.append(prompt)

        return prompts

    def _choose_instrumentation(self, emotions):
        """Choose appropriate instrumentation based on emotions"""
        if any(e in ["sadness", "tenderness", "calmness"] for e in emotions):
            return "piano and strings with subtle woodwinds"
        elif any(e in ["joy", "excitement"] for e in emotions):
            return "full orchestra with prominent brass and percussion"
        elif any(e in ["anger", "tension"] for e in emotions):
            return "distorted electric guitars, heavy percussion, and synthesizers"
        elif any(e in ["fear"] for e in emotions):
            return "dissonant strings, prepared piano, and electronic elements"
        else:
            return "chamber ensemble with piano, strings, and woodwinds"

    def _choose_style(self, emotions, scaled_params, batch_idx):
        """Choose musical style based on emotions and parameters"""
        tempo = scaled_params["tempo"][batch_idx].item()
        harmonic_complexity = scaled_params["harmonic_complexity"][batch_idx].item()

        if any(e in ["sadness", "tenderness"] for e in emotions) and tempo < 100:
            return "melancholic, cinematic"
        elif any(e in ["joy", "excitement"] for e in emotions) and tempo > 120:
            return "uplifting, energetic"
        elif any(e in ["fear", "tension"] for e in emotions):
            return "suspenseful, atmospheric"
        elif any(e in ["calmness"] for e in emotions):
            return "ambient, peaceful"
        elif any(e in ["anger"] for e in emotions):
            return "intense, dramatic"
        elif harmonic_complexity > 0.6:
            return "complex, avant-garde"
        else:
            return "melodic, contemporary"


# Function to integrate with MusicLM-style models
def generate_music_with_musiclm(text, emotion_extractor, emotion_mapper, musiclm_model=None):
    """
    Generate music from text using emotion mapping and MusicLM approach
    """
    # Extract emotions from text
    emotion_info = emotion_extractor(text)
    emotion_scores = emotion_info["emotion_scores"]
    emotion_categories = emotion_info["emotion_categories"]

    print(f"Extracted emotions with shape: {emotion_scores.shape}")

    # Get emotional arc for longer texts
    if len(text.split()) > 100:
        arc_info = emotion_extractor.extract_emotional_arc(text)
        emotional_arc = arc_info["emotional_arc"]
        segments = arc_info["segments"]
        print(f"Extracted emotional arc with shape: {emotional_arc.shape}")
    else:
        emotional_arc = None
        segments = [text]

    # Generate MusicLM prompts from emotions
    if emotional_arc is not None:
        # Generate multiple prompts for different segments
        prompts = emotion_mapper.generate_musiclm_prompt(emotional_arc, emotion_categories)
        print(f"Generated {len(prompts)} segment prompts from emotional arc")
    else:
        # Generate a single prompt
        prompts = emotion_mapper.generate_musiclm_prompt(emotion_scores, emotion_categories)
        print(f"Generated single prompt from emotion analysis")

    # Call MusicLM model if provided
    if musiclm_model is not None:
        # This would be the integration point with your MusicLM model
        print("Calling MusicLM model with generated prompts...")
        # Example: audio = musiclm_model.generate(prompts[0])
        audio = None  # Replace with actual MusicLM output
    else:
        audio = None

    return {
        "prompts": prompts,
        "emotion_scores": emotion_scores,
        "emotional_arc": emotional_arc,
        "segments": segments,
        "audio": audio
    }



class TemporalCoherenceModel(nn.Module):
    """Model for ensuring temporal coherence in music generation"""
    def __init__(self, music_param_dim=16, hidden_dim=128, num_layers=2):
        super(TemporalCoherenceModel, self).__init__()
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.music_param_dim = music_param_dim
        self.hidden_dim = hidden_dim

        # LSTM for modeling parameter sequences
        self.lstm = nn.LSTM(
            input_size=music_param_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )

        # Output layer to predict next parameters
        self.output_layer = nn.Linear(hidden_dim, music_param_dim)

        # Layer for embedding text condition into the LSTM
        self.text_condition_layer = nn.Linear(512, hidden_dim)  # Assuming text_embedding_dim=512
        self.to(self.device)

    def forward(self, param_sequence, text_embedding=None):
        """
        Process a sequence of musical parameters with optional text condition
        param_sequence: [batch_size, seq_length, music_param_dim]
        text_embedding: [batch_size, text_embedding_dim]
        """
        batch_size, seq_length = param_sequence.shape[0], param_sequence.shape[1]

        # Initialize hidden state, optionally with text condition
        h0 = torch.zeros(self.lstm.num_layers, batch_size, self.hidden_dim).to(param_sequence.device)
        c0 = torch.zeros(self.lstm.num_layers, batch_size, self.hidden_dim).to(param_sequence.device)

        if text_embedding is not None:
            text_hidden = self.text_condition_layer(text_embedding)
            h0[0] = text_hidden

        if segment_embedding is not None:
            # You'd need to add a segment embedding layer in __init__
            segment_hidden = self.segment_condition_layer(segment_embedding)
            # Combine with text embedding or use in second layer
            if text_embedding is not None:
                h0[0] = 0.7 * h0[0] + 0.3 * segment_hidden  # Weighted combination
            else:
                h0[0] = segment_hidden

        # Run LSTM
        lstm_out, (hn, cn) = self.lstm(param_sequence, (h0, c0))

        # Project to output parameter space
        output_sequence = self.output_layer(lstm_out)

        return output_sequence, (hn, cn)

    def generate_sequence(self, initial_params, sequence_length, text_embedding=None):
        """
        Generate a coherent sequence of musical parameters
        initial_params: [batch_size, music_param_dim]
        """
        """Generate a coherent sequence of musical parameters"""
        # Store current training state and set to eval mode
        was_training = self.training
        self.eval()

        try:
            batch_size = initial_params.shape[0]
            device = initial_params.device

            # Initialize sequence with initial parameters
            generated_sequence = [initial_params]

            # Initialize hidden state with text embedding if provided
            h = torch.zeros(self.lstm.num_layers, batch_size, self.hidden_dim).to(device)
            c = torch.zeros(self.lstm.num_layers, batch_size, self.hidden_dim).to(device)

            if text_embedding is not None:
                # Project text embedding to hidden dimension
                text_hidden = self.text_condition_layer(text_embedding)

                # Use text as initial hidden state in first layer
                h[0] = text_hidden

            # Generate sequence autoregressively
            current_params = initial_params

            for _ in range(sequence_length - 1):
                # Reshape for LSTM (add sequence dimension)
                params_input = current_params.unsqueeze(1)

                # Get prediction and update hidden state
                with torch.no_grad():
                    lstm_out, (h, c) = self.lstm(params_input, (h, c))
                    next_params = self.output_layer(lstm_out[:, -1, :])

                # Add to sequence
                generated_sequence.append(next_params)

                # Update current parameters
                current_params = next_params

                if smoothing > 0:
                    next_params = (1 - smoothing) * next_params + smoothing * current_params

            # Stack into a single tensor [batch_size, sequence_length, music_param_dim]
            return torch.stack(generated_sequence, dim=1)
        finally:
            # Restore previous training state
            self.train(was_training)

def generate_coherent_music_sequence(text, emotion_extractor, emotion_mapper, coherence_model, musiclm_model=None):
    """Generate coherent music across narrative segments"""

    # Extract emotional arc
    arc_info = emotion_extractor.extract_emotional_arc(text, num_segments=10)
    emotional_arc = arc_info["emotional_arc"]
    segments = arc_info["segments"]

    # Map emotions to initial music parameters
    music_params_sequence = []
    for i in range(emotional_arc.shape[0]):
        segment_emotions = emotional_arc[i]
        music_params = emotion_mapper.forward(segment_emotions)["music_params"]
        music_params_sequence.append(music_params)

    # Stack into sequence tensor
    music_params_sequence = torch.stack(music_params_sequence, dim=1)

    # Apply temporal coherence model to smooth transitions
    coherent_params = coherence_model.generate_sequence(
        music_params_sequence[:, 0, :],  # Initial parameters
        emotional_arc.shape[0],          # Sequence length
        text_embedding=None              # Could add text embedding if available
    )

    # Generate prompts from coherent parameters
    prompts = []
    for i in range(coherent_params.shape[1]):
        segment_params = coherent_params[:, i, :]
        # Convert parameters back to emotion space (you'd need to add this method)
        segment_emotions = emotion_mapper.params_to_emotions(segment_params)
        prompt = emotion_mapper.generate_musiclm_prompt(segment_emotions, arc_info["emotion_categories"])
        prompts.append(prompt[0])  # Assuming single batch

    # Now you could feed these prompts to MusicLM with appropriate transitions

    return {
        "coherent_params": coherent_params,
        "prompts": prompts,
        "segments": segments
    }

class EnhancedTextToMusicGenerationModel(nn.Module):
    """Enhanced text-to-music generation model with emotional analysis and temporal coherence"""
    def __init__(self, embedding_dim=512):
        super(EnhancedTextToMusicGenerationModel, self).__init__()

        # Text encoders
        self.text_encoder = MuLANTextEncoder(embedding_dim=embedding_dim)
        self.emotion_extractor = EmotionExtractionModule(num_emotions=8)

        # Emotion to music mapping
        self.emotion_mapper = EmotionToMusicMapper(emotion_dim=8, music_param_dim=16)

        # Temporal coherence model
        self.temporal_model = TemporalCoherenceModel(music_param_dim=16, hidden_dim=128)

        # Audio encoder for conditioning
        self.audio_encoder = SoundStreamEncoder(embedding_dim=embedding_dim)

        # Audio decoder for reconstruction
        self.audio_decoder = SoundStreamDecoder(embedding_dim=embedding_dim)

        # UNet for high-resolution generation
        self.unet = MusicConditionedUNet(embedding_dim=embedding_dim)

        # Music parameter conditioning layer (to inject into UNet)
        self.music_param_conditioning = nn.Linear(16, embedding_dim)

    def analyze_text(self, text, extract_arc=False):
        """Analyze text for emotional content and create music parameters"""
        # Get base text embedding
        text_embedding = self.text_encoder(text)

        # Extract emotional information
        if extract_arc:
            # For longer texts, extract emotional arc
            emotion_info = self.emotion_extractor.extract_emotional_arc(text)
            emotion_scores = emotion_info["emotional_arc"][0]  # Use first segment for initial params
            emotional_arc = emotion_info["emotional_arc"]
        else:
            # For shorter texts, get overall emotion
            emotion_info = self.emotion_extractor(text)
            emotion_scores = emotion_info["emotion_scores"]
            emotional_arc = None

        # Map emotions to musical parameters
        music_params = self.emotion_mapper(emotion_scores)

        # Generate temporal sequence of parameters if we have an emotional arc
        if emotional_arc is not None:
            # Generate parameter sequence for each segment in the arc
            param_sequences = []
            for i in range(emotional_arc.size(0)):
                segment_params = self.emotion_mapper(emotional_arc[i].unsqueeze(0))
                param_sequences.append(segment_params["music_params"])

            # Stack into sequence [sequence_length, batch_size, param_dim]
            param_sequence = torch.stack(param_sequences)

            # Apply temporal coherence
            smoothed_sequence = self.temporal_model.generate_sequence(
                param_sequences[0],
                len(param_sequences),
                text_embedding
            )

            # Also create a conditioning embedding from musical parameters
            music_conditioning = self.music_param_conditioning(music_params["music_params"])
            combined_embedding = text_embedding + 0.3 * music_conditioning
        else:
            param_sequence = None
            smoothed_sequence = None

            # Create conditioning embedding from musical parameters
            music_conditioning = self.music_param_conditioning(music_params["music_params"])
            combined_embedding = text_embedding + 0.3 * music_conditioning

        return {
            "text_embedding": text_embedding,
            "emotion_scores": emotion_scores,
            "music_params": music_params,
            "param_sequence": smoothed_sequence,
            "combined_embedding": combined_embedding,
            "emotional_arc": emotional_arc
        }

    def generate_from_text(self, text, noise_level=0.9, steps=50, length=16000, extract_arc=True):
        """
        Generate audio conditioned on text with enhanced emotional mapping
        """
        # Determine device
        device = next(self.parameters()).device

        # Analyze text for emotions and musical parameters
        analysis = self.analyze_text(text, extract_arc=extract_arc)
        print(f"Text analyzed. Emotion scores: {analysis['emotion_scores'].tolist()}")

        # Print key musical parameters for transparency
        scaled_params = analysis['music_params']['scaled_params']
        print(f"Musical parameters derived from text:")
        for param, value in scaled_params.items():
            if isinstance(value, torch.Tensor):
                print(f"  {param}: {value.item():.2f}")

        # Use combined embedding that incorporates musical parameters
        combined_embedding = analysis['combined_embedding'].to(device)
        print(f"Combined embedding shape: {combined_embedding.shape}")

        # Create initial noise
        audio = torch.randn(1, 1, length, device=device) * noise_level
        print(f"Initial noise created with shape: {audio.shape}")

        # Diffusion sampling loop with musical parameter conditioning
        with torch.no_grad():
            for i in range(steps):
                # Get model prediction
                update = self.unet(audio, combined_embedding)

                # Calculate noise scale for this step
                noise_scale = noise_level * (1.0 - (i / steps))

                # Make sure dimensions match before combining
                if audio.shape[2] != update.shape[2]:
                    min_len = min(audio.shape[2], update.shape[2])
                    audio = audio[:, :, :min_len]
                    update = update[:, :, :min_len]

                # Apply the update with noise scheduling
                audio = audio * (1.0 - noise_scale) + update * noise_scale

                # Optional: Print progress
                if i % 10 == 0:
                    print(f"Step {i}/{steps}, audio range: {audio.min().item():.3f} to {audio.max().item():.3f}")

                # Free up memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Normalize final audio to [-1, 1] range
        audio = torch.clamp(audio, -1.0, 1.0)

        return audio.cpu().numpy(), analysis

class MusicGenerationEvaluator:
    """Evaluation metrics for the text-to-music generation model"""
    def __init__(self):
        # Import libraries for audio analysis
        import librosa
        self.librosa = librosa

    def calculate_metrics(self, audio_array, sample_rate=16000):
        """Calculate objective metrics for generated audio"""
        # Convert to mono if needed
        if len(audio_array.shape) > 1 and audio_array.shape[0] > 1:
            audio_mono = audio_array.mean(axis=0)
        else:
            audio_mono = audio_array.squeeze()

        # Spectral features
        spectral_centroid = self.librosa.feature.spectral_centroid(y=audio_mono, sr=sample_rate)[0]
        spectral_contrast = self.librosa.feature.spectral_contrast(y=audio_mono, sr=sample_rate)[0]

        # Rhythm features
        tempo, _ = self.librosa.beat.beat_track(y=audio_mono, sr=sample_rate)

        # Harmonic-percussive separation
        harmonic, percussive = self.librosa.effects.hpss(audio_mono)

        # MFCC features
        mfccs = self.librosa.feature.mfcc(y=audio_mono, sr=sample_rate, n_mfcc=13)

        # Calculate statistics
        metrics = {
            "tempo": tempo,
            "spectral_centroid_mean": np.mean(spectral_centroid),
            "spectral_centroid_std": np.std(spectral_centroid),
            "spectral_contrast_mean": np.mean(spectral_contrast),
            "harmonic_ratio": np.sum(np.abs(harmonic)) / (np.sum(np.abs(percussive)) + 1e-8),
            "mfcc_means": np.mean(mfccs, axis=1).tolist(),
            "mfcc_stds": np.std(mfccs, axis=1).tolist(),
            "rms_energy": np.sqrt(np.mean(audio_mono**2)),
            "zero_crossing_rate": np.mean(self.librosa.feature.zero_crossing_rate(audio_mono)[0])
        }

        return metrics

    def compare_with_target(self, generated_metrics, target_metrics):
        """Compare generated audio with target metrics"""
        comparison = {}

        # Compare numerical metrics
        for key in generated_metrics:
            if key in target_metrics:
                if isinstance(generated_metrics[key], list):
                    # For lists (like MFCCs), calculate mean absolute difference
                    comparison[key + "_diff"] = np.mean(np.abs(np.array(generated_metrics[key]) -
                                                             np.array(target_metrics[key])))
                else:
                    # For single values
                    comparison[key + "_diff"] = abs(generated_metrics[key] - target_metrics[key])

        return comparison

    def emotion_alignment_score(self, target_emotions, music_params, audio_metrics):
        """
        Calculate how well the generated music aligns with target emotions
        based on both specified parameters and extracted audio features
        """
        alignment_scores = {}

        # Map emotions to expected audio features
        emotion_feature_map = {
            "joy": {
                "tempo": 0.7,                # Higher tempo
                "spectral_centroid_mean": 0.7,  # Brighter sound
                "harmonic_ratio": 0.7,       # More harmonic content
                "rms_energy": 0.7            # More energy
            },
            "sadness": {
                "tempo": 0.3,                # Lower tempo
                "spectral_centroid_mean": 0.4,  # Darker sound
                "harmonic_ratio": 0.6,       # More harmonic content
                "rms_energy": 0.4            # Less energy
            },
            # Add maps for other emotions...
        }

        # Normalize audio metrics to 0-1 range for comparison
        normalized_metrics = {
            "tempo": min(1.0, max(0.0, audio_metrics["tempo"] / 180.0)),
            "spectral_centroid_mean": min(1.0, max(0.0, audio_metrics["spectral_centroid_mean"] / 5000.0)),
            "harmonic_ratio": min(1.0, max(0.0, audio_metrics["harmonic_ratio"] / 5.0)),
            "rms_energy": min(1.0, max(0.0, audio_metrics["rms_energy"] / 0.3))
        }

        # Calculate alignment scores for each emotion
        for emotion, strength in target_emotions.items():
            if emotion in emotion_feature_map and strength > 0.2:
                # Get expected features for this emotion
                expected = emotion_feature_map[emotion]

                # Calculate distance to expected features
                feature_distances = []
                for feature, value in expected.items():
                    if feature in normalized_metrics:
                        distance = 1.0 - abs(normalized_metrics[feature] - value)
                        feature_distances.append(distance)

                # Average the distances
                if feature_distances:
                    alignment_scores[emotion] = sum(feature_distances) / len(feature_distances)

        # Calculate overall alignment
        weighted_score = 0
        total_weight = 0

        for emotion, strength in target_emotions.items():
            if emotion in alignment_scores:
                weighted_score += alignment_scores[emotion] * strength
                total_weight += strength

        if total_weight > 0:
            overall_score = weighted_score / total_weight
        else:
            overall_score = 0

        return {
            "emotion_scores": alignment_scores,
            "overall_alignment": overall_score
        }

# Missing encoder/decoder components
class MuLANTextEncoder(nn.Module):
    """Text encoder for music generation based on MuLAN architecture"""
    def __init__(self, embedding_dim=512, model_name="roberta-base"):
        super(MuLANTextEncoder, self).__init__()

        # Use RoBERTa as base model
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.text_model = RobertaModel.from_pretrained(model_name)

        # Project to specified embedding dimension
        self.projection = nn.Linear(self.text_model.config.hidden_size, embedding_dim)

    def forward(self, text):
        """
        Encode text into a fixed-size embedding
        text: string or list of strings
        """
        if isinstance(text, str):
            text = [text]

        # Tokenize text
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.text_model.device) for k, v in inputs.items()}

        # Get text embeddings
        with torch.no_grad():
            outputs = self.text_model(**inputs)

        # Use [CLS] token embedding as text representation
        text_embedding = outputs.last_hidden_state[:, 0, :]

        # Project to target dimension
        projected_embedding = self.projection(text_embedding)

        return projected_embedding


class SoundStreamEncoder(nn.Module):
    """Audio encoder based on SoundStream architecture"""
    def __init__(self, embedding_dim=512):
        super(SoundStreamEncoder, self).__init__()

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),

            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),  # Downsampling
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),

            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3),  # Downsampling
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),

            nn.Conv1d(128, 256, kernel_size=7, stride=2, padding=3),  # Downsampling
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),

            nn.Conv1d(256, embedding_dim, kernel_size=7, stride=2, padding=3),  # Downsampling
            nn.BatchNorm1d(embedding_dim),
            nn.LeakyReLU(0.1),
        )

        # Global average pooling for fixed-length embedding
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, audio):
        """
        Encode audio into a fixed-size embedding
        audio: [batch_size, channels, samples]
        """
        # Apply conv layers
        x = self.conv_layers(audio)

        # Global pooling
        embedding = self.global_pool(x).squeeze(-1)

        return embedding


class SoundStreamDecoder(nn.Module):
    """Audio decoder based on SoundStream architecture"""
    def __init__(self, embedding_dim=512):
        super(SoundStreamDecoder, self).__init__()

        # Initial projection
        self.initial_proj = nn.Linear(embedding_dim, embedding_dim * 4)

        # Reshape to [batch, channels, time]
        self.reshape = lambda x: x.view(x.size(0), embedding_dim, 4)

        # Transposed convolutional layers for upsampling
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose1d(embedding_dim, 256, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),

            nn.ConvTranspose1d(256, 128, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),

            nn.ConvTranspose1d(128, 64, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),

            nn.ConvTranspose1d(64, 32, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),

            nn.ConvTranspose1d(32, 1, kernel_size=7, stride=1, padding=3),
            nn.Tanh()  # Output in range [-1, 1]
        )

    def forward(self, embedding, length=16000):
        """
        Decode embedding to audio waveform
        embedding: [batch_size, embedding_dim]
        length: target audio length
        """
        # Initial projection and reshape
        x = self.initial_proj(embedding)
        x = x.view(x.size(0), -1, 4)  # [batch, channels, time]
        x = x.to(device)

        # Apply transposed convolutions
        audio = self.deconv_layers(x)

        # Ensure output has the right length
        if audio.shape[2] < length:
            # Pad if too short
            padding = torch.zeros(audio.shape[0], audio.shape[1], length - audio.shape[2],
                                device=audio.device)
            audio = torch.cat([audio, padding], dim=2)
        elif audio.shape[2] > length:
            # Truncate if too long
            audio = audio[:, :, :length]

        return audio


class MusicConditionedUNet(nn.Module):
    """UNet-based architecture for music generation with conditioning"""
    def __init__(self, embedding_dim=512, channels=[32, 64, 128, 256, 512]):
        super(MusicConditionedUNet, self).__init__()

        # Encoder blocks (downsampling path)
        self.encoder_blocks = nn.ModuleList()
        in_channels = 1  # Audio input channels

        for c in channels:
            self.encoder_blocks.append(self._make_encoder_block(in_channels, c))
            in_channels = c

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv1d(channels[-1], channels[-1] * 2, kernel_size=7, padding=3),
            nn.BatchNorm1d(channels[-1] * 2),
            nn.LeakyReLU(0.1),
            nn.Conv1d(channels[-1] * 2, channels[-1], kernel_size=7, padding=3),
            nn.BatchNorm1d(channels[-1]),
            nn.LeakyReLU(0.1)
        )

        # Conditioning projection
        self.cond_projection = nn.Sequential(
            nn.Linear(embedding_dim, channels[-1]),
            nn.LeakyReLU(0.1)
        )

        # Decoder blocks (upsampling path)
        self.decoder_blocks = nn.ModuleList()
        in_channels = channels[-1]

        for c in reversed(channels[:-1]):
            self.decoder_blocks.append(self._make_decoder_block(in_channels, c))
            in_channels = c

        # Final output layer
        self.final_layer = nn.Sequential(
            nn.Conv1d(channels[0], 1, kernel_size=7, padding=3),
            nn.Tanh()  # Output in range [-1, 1]
        )

    def _make_encoder_block(self, in_channels, out_channels):
        """Create a single encoder block"""
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.1),
            nn.Conv1d(out_channels, out_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.1)
        )

    def _make_decoder_block(self, in_channels, out_channels):
        """Create a single decoder block"""
        return nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.1),
            nn.Conv1d(out_channels, out_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.1)
        )

    def forward(self, audio, conditioning):
        """
        Forward pass through UNet with conditioning
        audio: [batch_size, channels, samples]
        conditioning: [batch_size, embedding_dim]
        """
        # Store skip connections
        skip_connections = []

        # Encoder path
        x = audio
        for encoder in self.encoder_blocks:
            x = encoder(x)
            skip_connections.append(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Apply conditioning
        cond = self.cond_projection(conditioning)  # [batch, channels[-1]]
        cond = cond.unsqueeze(-1)  # [batch, channels[-1], 1]
        cond = cond.expand(-1, -1, x.size(-1))  # [batch, channels[-1], time]

        # Add conditioning (residual connection)
        x = x + 0.3 * cond

        # Decoder path with skip connections
        for i, decoder in enumerate(self.decoder_blocks):
            skip = skip_connections[-(i+1)]

            # Ensure dimensions match before concatenating
            if x.shape[2] != skip.shape[2]:
                x = torch.nn.functional.interpolate(x, size=skip.shape[2], mode='linear')

            x = decoder(x + 0.1 * skip)  # Residual connection with skip

        # Final layer
        output = self.final_layer(x)

        return output

# Example main function focusing on MusicLM integration
def main_musiclm_integration():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize emotion components only (not the full music generation model)
    emotion_extractor = EmotionExtractionModule(num_emotions=8)
    emotion_mapper = EmotionToMusicMapper(emotion_dim=8, music_param_dim=16)

    # Move to device
    emotion_extractor = emotion_extractor.to(device)
    emotion_mapper = emotion_mapper.to(device)

    # Example literary text
    literary_text = """
    The old mansion stood silent against the stormy sky. Inside, memories of
    laughter and dance echoed through empty halls. A lone figure stood at the
    window, watching raindrops trace patterns like tears upon the glass.
    For years this place had been home, but tomorrow it would belong to strangers.
    """

    print("Analyzing text and generating MusicLM prompts...")
    print(f"Input text: {literary_text[:100]}...")

    # Generate prompts for MusicLM
    result = generate_music_with_musiclm(literary_text, emotion_extractor, emotion_mapper)

    # Print the generated prompts
    print("\nGenerated MusicLM Prompts:")
    for i, prompt in enumerate(result["prompts"]):
        print(f"\nPrompt {i+1}:")
        print(prompt)

    print("\nThese prompts can now be used with a MusicLM model to generate the actual music.")

    # Print emotional analysis
    print("\nEmotional Analysis:")
    for i, score in enumerate(result["emotion_scores"][0]):
        print(f"  {emotion_extractor.emotion_categories[i]}: {score.item():.3f}")

if __name__ == "__main__":
    main_musiclm_integration()