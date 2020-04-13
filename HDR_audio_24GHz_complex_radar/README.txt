HDR radar:

Was this the world's first HDR radar?

The 24GHz complex-valued Doppler radar datasets are text files with 8 columns:
First four columns are the real part, and the 2nd four columns, the imaginary.
The 1st and 5th column are captured with a gain of 10*,
the 2nd and 6th with a gain of 100*,
the 3rd and 7th with a gain of 1000*, and
the 4th and 8th with a gain of 10000*.

The four complex-valued gains correspond to coupling and cutoff as follows:

    10* DC-coupled;
    100* AC-coupled with cutoff 10/(2*pi) cycles per second;
    1000* AC-coupled with cutoff 100/(2*pi) cycles per second;
    10000* AC-coupled with cutoff 1000/(2*pi) cycles per second.

Here's some data to test chirplet transform and warblet transform:
dropping_1_object.txt = metal cover from computer fan dropped from 2m
dropping_2_objects.txt = wood circle dropped from 1m and fan cover from 2m
dropping_digikey_resistor_bags.txt = bags of resistors dropped
drop_resistor_tape.txt = resistors in tape roll form unrolled then dropped
ruler_8_inches.txt = plucking steel ruler with 8 inches protruding
ruler_16_inches.txt = plucking steel ruler with 16 inches protruding
spin_warblet.txt = spinning object (metal jar lid taped to wire)

Here's some bio-sensing data:
kyle_heart_active.txt
kyle_heart_baseline.txt
steve_heart_after_running1.txt = went for a short run outside
steve_heart_after_running2.txt = longer recording after running on the spot
two_hands.txt = one person waving at radar
four_hands.txt = two people waving at radar

Here's some data that might help with calibration and background analysis:
roomtone.txt = empty room late at night with no traffic on street outside

Data Analysis:

First step is Choleski factorization of covariance matrix as outlined on Page 38 of the course textbook, "Intelligent Image Processing", John Wiley and Sons.
Here is an excerpt: http://wearcam.org/HDR_audio_24GHz_complex_radar/chapter2excerpt.pdf

Carefully read Page 32 to 40, and try to understand the fundamental scientific concepts.

Next, you can easily reproduce the results shown on Page 2755 of this paper:
http://wearcam.org/chirplet.pdf
i.e. spectrogram of the sound from the falling object, and then the two falling objects...

Take a look at the spectrograms from all the datasets and try to understand what's happening.

Next you can do the chirplet transform on the HDR data.

Chirplet transform with HDR audio is a new window into the world of what's around us.

Try also simply listening on stereo headphones with one ear to the real part and the other ear to the imaginary part, before and after calibration with the Choleski factorization to listen to the spatialization and then gain insight into the system dynamics of radar as a sensor.

Next is to pick up on warblets and warblet transform,
https://ieeexplore.ieee.org/document/118914
and then throw in the machine learning with LEM (Mann) and, finally, once you've calibrated the data and familiarized yourself with it, try MPLEM (Richard Cui).

