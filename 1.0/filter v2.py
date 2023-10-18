import re
import json


def create_format_blacklist():
    # The following formats are in Australian format
    driver_license_pattern = r"\b\d{8}\b"
    passport_pattern = r"\b[A-Z]\d{7}\b"
    credit_card_pattern = r"\b\d{4} ?\d{4} ?\d{4} ?\d{4}\b"
    phone_number_pattern = r"\d{4} ?\d{3} ?\d{3}"
    phone_number_country_prefix_pattern = r"\+?\d{2} ?\d{3} ?\d{3} ?\d{3}"
    home_phone_number_pattern = r"\(?\d{2}\)? ?\d{4} ?\d{4}"
    tfn_pattern = r"\b\d{3} ?\d{3} ?\d{3}\b"
    crn_pattern = r"\b\d{3} ?\d{3} ?\d{3}[A-Z]\b"
    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,4}\b"
    home_address_pattern = r"\b\d+\s+[A-Za-z\s]+(?:Rd|St|Ave|Blvd|Dr|Ln|Ct)\b"

    pattern = f"({driver_license_pattern}|{passport_pattern}|{credit_card_pattern}|{phone_number_pattern}|{phone_number_country_prefix_pattern}|{home_phone_number_pattern}|{tfn_pattern}|{crn_pattern}|{email_pattern}|{home_address_pattern})"

    return re.compile(pattern)


def create_word_blacklist():
    # Contains offensive words
    blacklist = ["nigga", "faggot", "fuck", "fucking", "shit", "shitting", "crap",
                 "bastard", "cock", "penis", "vagina", "asshole", "bitch", "cunt",
                 "wanker", "motherfucker", "slut", "turd", "anal", "anus", "arse",
                 "blowjob", "ballsack", "biatch", "bloody", "bollock", "blow job",
                 "bollok", "boner", "boob", "butt", "buttplug", "clitoris", "dildo",
                 "hell", "nigger", "prick", "whore", "wtf", "jizz", "smegma", "tit"
                 "sex"]

    return blacklist


def filter_json_file(input_file, output_file, num_blacklist, word_blacklist):
    with open(input_file, 'r') as f_input, open(output_file, 'w') as f_output:
        data = json.load(f_input)
        cleaned_data = filter_json_objects(data, num_blacklist, word_blacklist)
        json.dump(cleaned_data, f_output, indent=2)


def filter_json_objects(data, num_blacklist, word_blacklist):
    if isinstance(data, list):
        cleaned_list = []
        for item in data:
            message = item.get("message", "")
            response = item.get("response", "")

            if not (
                re.search(num_blacklist, message) or
                re.search(num_blacklist, response) or
                any(word in message for word in word_blacklist) or
                any(word in response for word in word_blacklist)
            ):
                cleaned_list.append(item)
        return cleaned_list
    else:
        return data


if __name__ == "__main__":
    input_file = "input.json"
    output_file = "filtered.json"
    num_blacklist = create_format_blacklist()
    word_blacklist = create_word_blacklist()
    filter_json_file(input_file, output_file, num_blacklist, word_blacklist)
