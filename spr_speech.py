# spr_speech.py (Corrected Version)
import speech_recognition as sr
import time

def speech_module():
    """Enhanced speech recognition with better error handling and status feedback"""
    r = sr.Recognizer()
    r.energy_threshold = 300
    r.pause_threshold = 0.8
    
    try:
        with sr.Microphone() as source:
            print("Adjusting for ambient noise... Please wait.")
            r.adjust_for_ambient_noise(source, duration=1)
            print("Listening... Speak now!")
            
            audio_text = r.listen(source, timeout=10, phrase_time_limit=10)
            print("Processing speech...")
            
            try:
                text = r.recognize_google(audio_text)
                print(f"Recognized: {text}")
                return text
            except sr.UnknownValueError:
                return "Sorry, I could not understand the audio"
            except sr.RequestError as e:
                return f"Error with speech recognition service: {e}"
                
    except sr.WaitTimeoutError:
        return "No speech detected within timeout period"
    except Exception as e:
        return f"Microphone error: {str(e)}"

# In spr_speech.py

def continuous_listening(stop_event, stop_phrase="stop listening"):
    """
    Acts as a generator, yielding recognized text until a stop_event is set.
    Correctly handles the stop phrase to yield text captured just before it.
    """
    r = sr.Recognizer()
    r.energy_threshold = 400
    r.dynamic_energy_threshold = True
    
    with sr.Microphone() as source:
        print("Adjusting for ambient noise...")
        r.adjust_for_ambient_noise(source, duration=0.5)
        print(f"Continuous listening started. Say '{stop_phrase}' to end.")
        
        while not stop_event.is_set():
            try:
                audio = r.listen(source, timeout=1.5, phrase_time_limit=10)
                text = r.recognize_google(audio).lower()
                print(f"Recognized fragment: '{text}'")
                
                # Check if the stop phrase is in the recognized text
                if stop_phrase in text:
                    # Extract text that occurred *before* the stop phrase
                    pre_stop_text = text.split(stop_phrase, 1)[0].strip()
                    if pre_stop_text:
                        print(f"Yielding final fragment before stop: '{pre_stop_text}'")
                        yield pre_stop_text
                        
                    print("Stop phrase detected. Stopping listening.")
                    stop_event.set() # Signal the loop to stop
                    break # Exit the generator loop
                else:
                    # If no stop phrase, yield the whole fragment
                    yield text
                    
            except sr.WaitTimeoutError:
                continue # No speech, just keep listening
            except sr.UnknownValueError:
                continue # Speech unintelligible
            except sr.RequestError as e:
                print(f"API Error: {e}")
                break
            except Exception as e:
                print(f"An unexpected error occurred in generator: {e}")
                break
    
    print("Continuous listening generator finished.")