def GET_STAR_TEMP(sptype):
    temperature = 0 

    if sptype.startswith("sd"):
        sptype = sptype[2:]

    if sptype.startswith("O3"):
        temperature = 5
    elif sptype.startswith("O4"):
        temperature = 7
    elif sptype.startswith("O5"):
        temperature = 10
    elif sptype.startswith("O6"):
        temperature = 11
    elif sptype.startswith("O7"):
        temperature = 13
    elif sptype.startswith("O8"):
        temperature = 15
    elif sptype.startswith("O9"):
        temperature = 18
    elif sptype.startswith("B0"):
        temperature = 20
    elif sptype.startswith("B1"):
        temperature = 25
    elif sptype.startswith("B2"):
        temperature = 28
    elif sptype.startswith("B3"):
        temperature = 31
    elif sptype.startswith("B4"):
        temperature = 33
    elif sptype.startswith("B5"):
        temperature = 35
    elif sptype.startswith("B6"):
        temperature = 36
    elif sptype.startswith("B7"):
        temperature = 37
    elif sptype.startswith("B8"):
        temperature = 41
    elif sptype.startswith("B9"):
        temperature = 49
    elif sptype.startswith("A0"):
        temperature = 51
    elif sptype.startswith("A1"):
        temperature = 52
    elif sptype.startswith("A2"):
        temperature = 53
    elif sptype.startswith("A3"):
        temperature = 56
    elif sptype.startswith("A4"):
        temperature = 56
    elif sptype.startswith("A5"):
        temperature = 56
    elif sptype.startswith("A6"):
        temperature = 57
    elif sptype.startswith("A7"):
        temperature = 58
    elif sptype.startswith("A8"):
        temperature = 59
    elif sptype.startswith("A9"):
        temperature = 60
    elif sptype.startswith("F0"):
        temperature = 60
    elif sptype.startswith("F1"):
        temperature = 63
    elif sptype.startswith("F2"):
        temperature = 61
    elif sptype.startswith("F3"):
        temperature = 62
    elif sptype.startswith("F4"):
        temperature = 62
    elif sptype.startswith("F5"):
        temperature = 63
    elif sptype.startswith("F6"):
        temperature = 63
    elif sptype.startswith("F7"):
        temperature = 63
    elif sptype.startswith("F8"):
        temperature = 64
    elif sptype.startswith("F9"):
        temperature = 65
    elif sptype.startswith("G0"):
        temperature = 65
    elif sptype.startswith("G1"):
        temperature = 66
    elif sptype.startswith("G2"):
        temperature = 66
    elif sptype.startswith("G3"):
        temperature = 66
    elif sptype.startswith("G4"):
        temperature = 66
    elif sptype.startswith("G5"):
        temperature = 66
    elif sptype.startswith("G6"):
        temperature = 66
    elif sptype.startswith("G7"):
        temperature = 67
    elif sptype.startswith("G8"):
        temperature = 67
    elif sptype.startswith("G9"):
        temperature = 67
    elif sptype.startswith("K0"):
        temperature = 68
    elif sptype.startswith("K1"):
        temperature = 69
    elif sptype.startswith("K2"):
        temperature = 70
    elif sptype.startswith("K3"):
        temperature = 70
    elif sptype.startswith("K4"):
        temperature = 71
    elif sptype.startswith("K5"):
        temperature = 72
    elif sptype.startswith("K6"):
        temperature = 72
    elif sptype.startswith("K7"):
        temperature = 73
    elif sptype.startswith("K8"):
        temperature = 73
    elif sptype.startswith("K9"):
        temperature = 73
    elif sptype.startswith("M0"):
        temperature = 74
    elif sptype.startswith("M1"):
        temperature = 74
    elif sptype.startswith("M2"):
        temperature = 75
    elif sptype.startswith("M3"):
        temperature = 75
    elif sptype.startswith("M4"):
        temperature = 75
    elif sptype.startswith("M5"):
        temperature = 75
    elif sptype.startswith("M6"):
        temperature = 75
    elif sptype.startswith("M7"):
        temperature = 75
    elif sptype.startswith("M8"):
        temperature = 75
    elif sptype.startswith("M9"):
        temperature = 75
    elif sptype.startswith("O"):
        temperature = 13
    elif sptype.startswith("B"):
        temperature = 35
    elif sptype.startswith("M"):
        temperature = 75
    elif sptype.startswith("C"):
        temperature = 75
    elif sptype.startswith("A"):
        temperature = 56
    elif sptype.startswith("R"):
        temperature = 75
    elif sptype.startswith("G"):
        temperature = 66
    elif sptype.startswith("W"):
        temperature = 35
    elif sptype.startswith("K"):
        temperature = 72
    elif sptype.startswith("N"):
        temperature = 72
    elif sptype.startswith("S"):
        temperature = 72
    elif sptype.startswith("F"):
        temperature = 63
    elif sptype.startswith("DA"):
        temperature = 35
    else:
        # Handle other spectral types or unknown cases
        temperature = 66  # Default temperature

    return temperature

# Example usage
# hipline = {"sp_type": "sdB3V", "temperature": 0}
# temperature = GET_STAR_TEMP(hipline)
# print(f"Temperature: {temperature}")
