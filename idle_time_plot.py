from PIL import Image, ImageDraw

if __name__ == '__main__':
    per_interval_width = 20
    height = 140

    times_1_3_constant = [(1668964174.352554, 1668964174.425982), (1668964174.425984, 1668964174.4260101), (1668964174.4385228, 1668964174.4385521), (1668964174.483126, 1668964174.539134), (1668964174.539136, 1668964174.53916), (1668964174.576701, 1668964174.576756), (1668964174.626792, 1668964174.6855738), (1668964174.685576, 1668964174.685599), (1668964174.720764, 1668964174.720801), (1668964174.768333, 1668964174.8256059), (1668964174.825608, 1668964174.8256278), (1668964174.859991, 1668964174.860029), (1668964174.9083898, 1668964174.965023), (1668964174.965025, 1668964174.965044), (1668964175.000448, 1668964175.000479), (1668964175.04775, 1668964175.105472), (1668964175.105474, 1668964175.105498), (1668964175.1430411, 1668964175.143073), (1668964175.1806002, 1668964175.238234), (1668964175.238235, 1668964175.238255), (1668964175.275178, 1668964175.27522), (1668964175.322486, 1668964175.3798661), (1668964175.3798678, 1668964175.379888), (1668964175.4283872, 1668964175.428422), (1668964175.474914, 1668964175.5323908), (1668964175.532394, 1668964175.532421), (1668964175.568995, 1668964175.569026), (1668964175.605922, 1668964175.66321), (1668964175.663212, 1668964175.663232), (1668964175.7087538, 1668964175.7087831), (1668964175.754826, 1668964175.813783), (1668964175.813785, 1668964175.8138082), (1668964175.8514, 1668964175.851437), (1668964175.888971, 1668964175.9471412), (1668964175.947144, 1668964175.947165), (1668964175.9827871, 1668964175.982827), (1668964176.032871, 1668964176.0901701), (1668964176.0901709, 1668964176.0901911), (1668964176.125262, 1668964176.125298), (1668964176.1745641, 1668964176.2317162), (1668964176.2317178, 1668964176.23174), (1668964176.269264, 1668964176.269294), (1668964176.3268511, 1668964176.383787)]
    times_9_10_constant = [(1668968739.443605, 1668968739.498875), (1668968739.498882, 1668968739.498923), (1668968739.498924, 1668968739.5554922), (1668968739.555499, 1668968739.61104), (1668968739.6110458, 1668968739.668304), (1668968739.66831, 1668968739.724312), (1668968739.724318, 1668968739.7776399), (1668968739.777646, 1668968739.8336132), (1668968739.833622, 1668968739.889448), (1668968739.8894541, 1668968739.943603), (1668968739.9436102, 1668968739.99996), (1668968739.9999669, 1668968739.9999979), (1668968739.999999, 1668968740.0579998), (1668968740.058006, 1668968740.113461), (1668968740.113466, 1668968740.1693618), (1668968740.169368, 1668968740.2250612), (1668968740.225066, 1668968740.280421), (1668968740.280427, 1668968740.33664), (1668968740.33665, 1668968740.3935318), (1668968740.393538, 1668968740.449431), (1668968740.4494379, 1668968740.503644), (1668968740.50365, 1668968740.50368), (1668968740.503681, 1668968740.560311), (1668968740.560317, 1668968740.6152298), (1668968740.6152372, 1668968740.671061), (1668968740.671068, 1668968740.728765), (1668968740.7287722, 1668968740.784584), (1668968740.784589, 1668968740.838763), (1668968740.838769, 1668968740.894801), (1668968740.894807, 1668968740.950181), (1668968740.950188, 1668968741.0054002), (1668968741.005405, 1668968741.005435), (1668968741.005436, 1668968741.062405), (1668968741.0624099, 1668968741.117249), (1668968741.117256, 1668968741.174274), (1668968741.17428, 1668968741.230667), (1668968741.230674, 1668968741.286784), (1668968741.28679, 1668968741.3436952), (1668968741.343704, 1668968741.400395), (1668968741.4004, 1668968741.457993), (1668968741.458, 1668968741.514879), (1668968741.5148852, 1668968741.514921), (1668968741.514922, 1668968741.571619), (1668968741.571625, 1668968741.627658)]

    times_dyn_1 = [ (1669055561.555187, 1669055561.611933), (1669055561.6119401, 1669055561.611964), (1669055561.649527, 1669055561.7051609), (1669055561.7051678, 1669055561.760486), (1669055561.760492, 1669055561.817109), (1669055561.8171148, 1669055561.817148), (1669055561.8421812, 1669055561.89777), (1669055561.8977768, 1669055561.9534771), (1669055561.953482, 1669055561.953522), (1669055561.9894521, 1669055562.045839), (1669055562.045845, 1669055562.100641), (1669055562.100645, 1669055562.100673), (1669055562.125864, 1669055562.1805198), (1669055562.180525, 1669055562.238077), (1669055562.2380831, 1669055562.238115), (1669055562.274637, 1669055562.333645), (1669055562.333652, 1669055562.3893209), (1669055562.389326, 1669055562.389365), (1669055562.412988, 1669055562.469366), (1669055562.4693718, 1669055562.5306711), (1669055562.530678, 1669055562.5871308), (1669055562.587137, 1669055562.587166), (1669055562.599705, 1669055562.6561239), (1669055562.656131, 1669055562.714954), (1669055562.7149599, 1669055562.772015), (1669055562.772021, 1669055562.772049), (1669055562.782392, 1669055562.838675), (1669055562.838683, 1669055562.893458), (1669055562.893462, 1669055562.948372), (1669055562.9483788, 1669055562.948415), (1669055562.972409, 1669055563.03013), (1669055563.030134, 1669055563.0846038), (1669055563.08461, 1669055563.138937), (1669055563.138943, 1669055563.138975), (1669055563.163017, 1669055563.221956), (1669055563.221962, 1669055563.280328), (1669055563.280336, 1669055563.335105), (1669055563.335111, 1669055563.33515), (1669055563.345431, 1669055563.400128), (1669055563.4001331, 1669055563.4556448), (1669055563.455654, 1669055563.540833), (1669055563.54084, 1669055563.540874), (1669055563.5408762, 1669055563.599327), (1669055563.59933, 1669055563.654706), (1669055563.6547148, 1669055563.710003), (1669055563.71001, 1669055563.7100391), (1669055563.735072, 1669055563.790803), (1669055563.7908108, 1669055563.849301), (1669055563.8493068, 1669055563.9043279), (1669055563.904336, 1669055563.9043689), (1669055563.916588, 1669055563.982602), (1669055563.9826062, 1669055564.038367), (1669055564.0383732, 1669055564.093371), (1669055564.093375, 1669055564.093405), (1669055564.105951, 1669055564.1788201), (1669055564.178826, 1669055564.2367308), (1669055564.236736, 1669055564.293244)]

    times_dyn_2 = [(1669057647.8673239, 1669057647.8673449), (1669057647.8673458, 1669057647.9486), (1669057647.948606, 1669057647.948632), (1669057647.9486332, 1669057648.03012), (1669057648.030127, 1669057648.0301492), (1669057648.03015, 1669057648.112396), (1669057648.1124039, 1669057648.112438), (1669057648.1124392, 1669057648.192344), (1669057648.19235, 1669057648.192373), (1669057648.192374, 1669057648.272547), (1669057648.272553, 1669057648.272576), (1669057648.272577, 1669057648.357227), (1669057648.357235, 1669057648.3572621), (1669057648.3572638, 1669057648.4403079), (1669057648.4403148, 1669057648.440338), (1669057648.44034, 1669057648.52526), (1669057648.525268, 1669057648.525295), (1669057648.525297, 1669057648.6101918), (1669057648.610197, 1669057648.610219), (1669057648.6102211, 1669057648.69026), (1669057648.6902661, 1669057648.690287), (1669057648.6902888, 1669057648.7735941), (1669057648.773601, 1669057648.773623), (1669057648.7736242, 1669057648.857745), (1669057648.85775, 1669057648.8577769), (1669057648.8577778, 1669057648.942125), (1669057648.942129, 1669057648.942167), (1669057648.942168, 1669057649.0271602), (1669057649.027167, 1669057649.027191), (1669057649.027192, 1669057649.112747), (1669057649.112758, 1669057649.112782), (1669057649.1127832, 1669057649.196521), (1669057649.196527, 1669057649.1965492), (1669057649.19655, 1669057649.275326), (1669057649.2753332, 1669057649.2753541), (1669057649.2753549, 1669057649.356907), (1669057649.3569138, 1669057649.356938), (1669057649.356939, 1669057649.439106), (1669057649.439111, 1669057649.43913), (1669057649.4391308, 1669057649.520492), (1669057649.520502, 1669057649.52053), (1669057649.520532, 1669057649.603242), (1669057649.6032481, 1669057649.6032681), (1669057649.6032698, 1669057649.68226), (1669057649.682267, 1669057649.682292), (1669057649.682293, 1669057649.76316), (1669057649.763168, 1669057649.763204), (1669057649.763205, 1669057649.8457952), (1669057649.8458061, 1669057649.845835), (1669057649.845836, 1669057649.926799), (1669057649.926805, 1669057649.926827), (1669057649.926829, 1669057650.009741), (1669057650.0097458, 1669057650.009769), (1669057650.00977, 1669057650.096125), (1669057650.0961332, 1669057650.096156), (1669057650.108703, 1669057650.192182), (1669057650.1921878, 1669057650.192211), (1669057650.192213, 1669057650.27474), (1669057650.274747, 1669057650.274774), (1669057650.286711, 1669057650.36776), (1669057650.367767, 1669057650.367789), (1669057650.3803022, 1669057650.461633), (1669057650.46164, 1669057650.461665), (1669057650.474211, 1669057650.550401), (1669057650.550406, 1669057650.5504289), (1669057650.5754728, 1669057650.657054), (1669057650.657061, 1669057650.657084), (1669057650.669596, 1669057650.750417), (1669057650.750429, 1669057650.750454), (1669057650.762995, 1669057650.847407), (1669057650.847413, 1669057650.847438), (1669057650.847439, 1669057650.926881)]

    times = times_dyn_2

    print(times[-1][1] - times[0][0])

    times_normal = []

    min_time = times[0][0]
    for index in range(len(times)):
        t1 = times[index][0] - min_time
        t2 = times[index][1] - min_time

        t1 = int(t1 * 1000000)
        t2 = int(t2 * 1000000)

        times_normal.append([t1, t2])

    print(times_normal)

    print(len(times))
    print(len(times_normal))

    times = times_normal

    # process_times = times.copy()
    # idle_times = []
    #
    # for index in range(len(times)):
    #     if index + 1 == len(times):
    #         continue
    #
    #     idle_times.append((times[index][1], times[index + 1][0]))

    min_max_width = int(times[-1][1] - times[0][0])
    image_width = 900

    width_per_1 = image_width / min_max_width

    print(width_per_1)

    im = Image.new('RGB', (image_width, height), color=(196, 198, 231))

    height -= 40

    draw = ImageDraw.Draw(im)
    draw.rectangle((0, 0, image_width, height), fill=(196, 198, 231), width=1, outline=(0, 0, 0))

    draw.rectangle((0, 100, image_width, height + 40), fill=(255, 255, 255), width=1, outline=(0, 0, 0))

    for time in times:
        draw = ImageDraw.Draw(im)
        width = time[1] * width_per_1 - time[0] * width_per_1
        if width > 1:
            draw.rectangle((time[0] * width_per_1, 0, time[1] * width_per_1, height), fill=(206, 179, 171), width=1, outline=(0, 0, 0))

    for time in times:
        draw = ImageDraw.Draw(im)
        width = time[1] * width_per_1 - time[0] * width_per_1
        if width < 1:
            draw.rectangle((time[0] * width_per_1, 0, time[1] * width_per_1 + 2, height), fill=(70, 99, 101))

    im.save("diagram.png", "PNG")
