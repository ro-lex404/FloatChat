# spr_speech.py (Corrected Version)
import speech_recognition as sr
import time

# Replace the function in spr_speech.py with this improved version

def continuous_listening(stop_event, stop_phrase="stop listening"):
    """
    A more resilient version of the listening generator.
    It continues listening even if there are temporary network/API errors.
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
                
                if stop_phrase in text:
                    pre_stop_text = text.split(stop_phrase, 1)[0].strip()
                    if pre_stop_text:
                        print(f"Yielding final fragment before stop: '{pre_stop_text}'")
                        yield pre_stop_text
                    print("Stop phrase detected. Stopping listening.")
                    stop_event.set()
                    break # Exit ONLY when the stop phrase is heard
                else:
                    yield text
                    
            except sr.WaitTimeoutError:
                # This is normal, just means no speech was detected.
                continue
            except sr.UnknownValueError:
                # Speech was detected but couldn't be understood.
                print("Could not understand audio, continuing to listen...")
                continue
            except sr.RequestError as e:
                # KEY FIX: Don't break on network errors, just report and continue.
                print(f"API Error: {e}. Retrying...")
                time.sleep(1) # Wait a second before trying again
                continue
            except Exception as e:
                # For any other unexpected error, it's safer to stop.
                print(f"An critical error occurred in generator: {e}")
                break
    
    print("Continuous listening generator finished.")