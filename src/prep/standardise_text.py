import re
import inflect

p = inflect.engine()
  
def convert_number_to_words_safe(number_str):
    """
    Converts a numerical string to its word representation safely.
    Handles potential ValueError for non-integer strings and other 
    exceptions.
    """

    try:
        words = p.number_to_words(int(number_str))
        return words.replace('-', ' ')
    except ValueError:
        return number_str
    except Exception as e:
        print(f"Error converting '{number_str}' to words: {e}")
        return number_str


def replace_bus_names(match):
    """
    Regex replacement function for patterns like "123A" (bus names).
    Converts the number part to words and appends the letter part.
    """
    num_part = match.group(1)
    letter_part = match.group(2)
    num_word = convert_number_to_words_safe(num_part)
    
    return f"{num_word} {letter_part}"


def replace_time_am_pm(match):
    """
    Regex replacement function for time patterns like "9am" or "5 p.m.".
    Converts the number part to words and spells out "am" or "pm".
    """
    num_part = match.group(1)
    ampm_part = match.group(2)
    num_word = convert_number_to_words_safe(num_part)
    
    spaced_ampm = " ".join(list(ampm_part))
    return f"{num_word} {spaced_ampm}"


def replace_time_colon(match):
    """
    Regex replacement function for time patterns like "10:30" or "8:00".
    Converts hour and minute parts to words. Handles "oh" for minutes 01-09.
    """
    hour_part = match.group(1)
    minute_part = match.group(2)
    
    hour_word = convert_number_to_words_safe(hour_part)

    if minute_part == '00':
        return hour_word
    elif 0 < int(minute_part) < 10:
        minute_word = f"oh {convert_number_to_words_safe(minute_part[1])}"
    else:
        minute_word = convert_number_to_words_safe(minute_part)
    
    return f"{hour_word} {minute_word}"


def replace_three_digit(match):
    """
    Regex replacement function for three-digit numbers.
    Converts numbers like "105" to "one oh five". Other three-digit numbers
    are converted normally (e.g., "123" to "one hundred twenty three").
    """
    num_str = match.group(0)
    first_digit = num_str[0]
    second_digit = num_str[1]
    third_digit = num_str[2]
    
    if second_digit == '0' and third_digit != '0':
        first_word = convert_number_to_words_safe(first_digit)
        third_word = convert_number_to_words_safe(third_digit)
        return f"{first_word} oh {third_word}"
    else:
        return convert_number_to_words_safe(num_str)

def process_text_for_alignment(text):
    """
    Processes text to produce three versions:
    1. Original raw text.
    2. Processed text for output (with number expansions).
    3. Text for matching (lowercased, number expansions, non-alphanumeric 
    chars removed).
    """
    if not isinstance(text, str):
        text = str(text)

    original_raw_text = text

    # Standardize spaces and remove newlines/tabs
    processed_text_for_output = (
        text.replace('\xa0', ' ').replace('\n', ' ').replace('\t', ' ')
    )

    # Apply bus name, time, and number to words transformations
    processed_text_for_output = re.sub(
        r'\b(\d+)([a-zA-Z]+)\b', 
        replace_bus_names, 
        processed_text_for_output
    )
    processed_text_for_output = re.sub(
        r'\b(\d+)\s*(a\.?m\.?|p\.?m\.?)\b', 
        replace_time_am_pm, 
        processed_text_for_output
    )
    processed_text_for_output = re.sub(
        r'\b(\d{1,2}):(\d{2})\b', 
        replace_time_colon, 
        processed_text_for_output
    )
    processed_text_for_output = re.sub(
        r'\b(\d{3})\b', 
        replace_three_digit, 
        processed_text_for_output
    )
    processed_text_for_output = re.sub(
        r'\b(\d+)\b', 
        lambda x: convert_number_to_words_safe(x.group(1)), 
        processed_text_for_output
    )
    
    # Clean up extra spaces in the output version
    processed_text_for_output = re.sub(
        r'\s+', ' ', processed_text_for_output
    ).strip()

    # Prepare the version for matching: lowercase and remove 
    # non-alphanumeric chars
    text_for_matching = processed_text_for_output.lower()
    text_for_matching = re.sub(r'[^a-z0-9\s]', ' ', text_for_matching)
    text_for_matching = re.sub(r'\s+', ' ', text_for_matching).strip()

    return original_raw_text, processed_text_for_output, text_for_matching
