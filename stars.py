from astropy.io import fits
import astropy.table as apy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import urllib
from IPython.display import Image, display
import io
import random
from scipy import interpolate
import colour
import colour.plotting as cplt
import base64
from astropy.coordinates import SkyCoord


class StarLoader:
    def __init__(
        self,
        starpath='data/mastarall-v3_1_1-v1_7_7.fits',
        spectrapath='data/mastar-combspec-v3_1_1-v1_7_7-lsfpercent99.5.fits',
        paramspath='data/mastar-goodstars-v3_1_1-v1_7_7-params-v1.fits',
        hdu=1,
    ):
        starsdat = apy.Table.read(starpath, format='fits', hdu=hdu)
        assert len(starsdat) > 10
        spectradat = apy.Table.read(spectrapath, format='fits', hdu=hdu)
        assert len(spectradat) > 10

        stars = apy.join(
            starsdat[['MANGAID', 'RA', 'DEC', 'INPUT_TEFF', 'INPUT_SOURCE']],
            spectradat[['MANGAID', 'WAVE', 'FLUX', 'IVAR', 'MASK']],
            'MANGAID'
        )

        if paramspath != '':
            paramsdat = apy.Table.read(paramspath, format='fits', hdu=hdu)
            assert len(paramsdat) > 10
            stars = apy.join(
                stars[['MANGAID', 'RA', 'DEC', 'INPUT_TEFF',
                       'INPUT_SOURCE', 'WAVE', 'FLUX', 'IVAR', 'MASK']],
                paramsdat[['MANGAID', 'TEFF_MED']],
                'MANGAID'
            )

        stars['FLUX_CORR'] = stars['FLUX'].copy()

        # Interpolate over broken pixels. Good example of test target is 7-17015390
        for s in stars:
            good_mask = ((s['IVAR'] != 0) & (s['MASK'] == 0))
            first_good_val = s['FLUX'][np.where(good_mask)[0][0]]
            last_good_val = s['FLUX'][np.where(good_mask)[0][-1]]
            good_indices = np.where(good_mask)[0]
            if len(good_indices) == len(s['FLUX']):
                # All points good
                continue

            if len(good_indices) == 0:
                # All points bad...
                continue

            fint = interpolate.interp1d(
                good_indices, s['FLUX'][good_mask], bounds_error=False, copy=False, fill_value=(first_good_val, last_good_val))
            s['FLUX_CORR'] = np.where(
                good_mask, s['FLUX'], fint(np.arange(len(s['FLUX']))))
            # Extrapolation can create nagatives. Clamp to 0.0
            s['FLUX_CORR'][s['FLUX_CORR'] < 0.0] = 0.0

        self.stars = stars

    def get_goodt(self):
        return self.stars[self.stars['INPUT_TEFF'] > 0]

    def get_badt(self):
        return self.stars[self.stars['INPUT_TEFF'] <= 0]

    def get_badt_lim(self):
        goodt = self.get_goodt()['INPUT_TEFF']
        max_seen = goodt.max()
        min_seen = goodt.min()
        return self.stars[(self.stars['INPUT_TEFF'] <= 0) & (self.stars['TEFF_MED'] >= min_seen) & (self.stars['TEFF_MED'] <= max_seen)]

    def fits_to_pd(self, stars):
        mangaid = pd.Series(stars['MANGAID'].astype('str')).rename('mangaid')
        teff = pd.Series(stars['INPUT_TEFF'].astype(float)).rename('teff')
        teff_ext = pd.Series(
            stars['TEFF_MED'].astype(float)).rename('teff_ext')

        header = []
        for c in np.arange(stars['FLUX_CORR'].shape[1]):
            header.append('flux%d' % c)
        flux_corr = pd.DataFrame(
            np.array(stars['FLUX_CORR'].astype(float)), columns=header)

        return pd.concat([mangaid, flux_corr, teff, teff_ext], axis=1)

    def get_angstroms(self):
        return np.array(self.stars[0]['WAVE'].astype(float))

    def get_star(self, mangaid):
        starq = self.stars[np.char.startswith(
            self.stars['MANGAID'].data, mangaid.encode('ascii'))]
        if len(starq) > 0:
            # BUG: first match, but matches using startswith, so it can match the wrong thing
            return Star(starq[0])
        else:
            raise Exception('Star not found')

    def random_star(self, mask=False):
        if not isinstance(mask, bool):
            s = self.stars[mask]
        else:
            s = self.stars

        return Star(s[random.randint(0, len(s)-1)])

    def locate_star(self, star):
        _, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(self.stars['RA'], self.stars['DEC'],
                   marker='o', color='black', alpha=0.1, s=1)
        ax.scatter([star.stardata['RA']], [star.stardata['DEC']],
                   marker='+', color='red', s=250)
        plt.show()

    def wrap_in_table(self, o):
        return '<table border=1><tr><th>MANGAID</th><th>Image</th><th>Spectrum</th><th>Input T</th><th>Input source</th></tr>%s</table>' % o


class Star:

    def __init__(self, stardata):
        self.stardata = stardata

        # Divide by 10.0 to convert from Ang to nm
        spectrum_data = dict(
            zip(self.stardata['WAVE']/10.0, self.stardata['FLUX_CORR']))
        self.spectral_distribution = colour.SpectralDistribution(spectrum_data)

        with colour.utilities.domain_range_scale('1'):
            self.xyz = colour.sd_to_XYZ(self.spectral_distribution.align(
                colour.SPECTRAL_SHAPE_DEFAULT))
            self.rgb = colour.XYZ_to_sRGB(
                self.xyz / 100, illuminant=colour.CCS_ILLUMINANTS['cie_2_1931']['E'])
            self.rgb = colour.algebra.normalise_maximum(self.rgb)

    def get_mangaid(self):
        return self.stardata['MANGAID'].strip()

    def get_coord(self):
        # You can .to_string('decimal|dms|hmsdms')
        return SkyCoord(self.stardata['RA'], self.stardata['DEC'], unit='deg')

    def get_star_image(self, w=50, h=50, s=0.2):
        # https://skyserver.sdss.org/dr17/SkyServerWS/ImgCutout/getjpeg?TaskName=Skyserver.Explore.Image&ra=48.8733428617884%20&dec=-8.44283369381076&scale=0.2&width=200&height=200&opt=G
        params = {
            'TaskName': 'Skyserver.Explore.Image',
            'ra': self.stardata['RA'],
            'dec': self.stardata['DEC'],
            'scale': s,
            'width': w,
            'height': h,
        }
        try:
            with urllib.request.urlopen('http://skyserver.sdss.org/dr17/SkyServerWS/ImgCutout/getjpeg?%s' % urllib.parse.urlencode(params)) as response:
                return response.read()
        except urllib.error.HTTPError as err:
            return b''

    def show_star_image(self):
        i = self.get_star_image(w=200, h=200, s=0.3)
        if len(i) > 0:
            display(Image(i))
        else:
            print('No image')

    def get_spectrum_thumbnail(self):
        plotdata = io.BytesIO()
        plt.figure(figsize=(2, 1))
        plt.axis('off')
        plt.plot(self.stardata['FLUX_CORR'])
        plt.savefig(plotdata, format='png')
        plt.close()

        return plotdata.getvalue()

    def show_spectrum(self):
        _, axflux = plt.subplots(figsize=(20, 5))

        # Plot spectrum, so that red uncorrected line is showing from under the interpolated one
        axflux.set_ylabel('FLUX (red=uncorrected)', color='navy')
        axflux.ticklabel_format(style='plain')
        axflux.plot(self.stardata['WAVE'],
                    self.stardata['FLUX'], color='red', linewidth=0.5)
        axflux.plot(
            self.stardata['WAVE'], self.stardata['FLUX_CORR'], color='navy', linewidth=1.0)

        # Inverse variances
        axivar = axflux.twinx()
        axivar.ticklabel_format(style='plain')
        axivar.set_ylabel('IVAR', color='orange')
        axivar.scatter(self.stardata['WAVE'], self.stardata['IVAR'],
                       color='orange', alpha=0.2, marker='o', s=0.5)

        # Mix together ivar=0 and mask>0 - these have special meaning, apparently highlighting "bad pixels"
        good_mask = ((self.stardata['IVAR'] != 0) &
                     (self.stardata['MASK'] == 0))
        axmask = axflux.twinx()
        axmask.set_ylabel('GOOD_MASK (gray=bad pixel)', color='gray')
        axmask.set_ylim(0.0, 0.5)
        axmask.spines['right'].set_position(('axes', 1.10))
        axmask.vlines(self.stardata['WAVE'], good_mask,
                      color='gray', alpha=0.1, ymax=1.0)

        plt.show()

    def get_spectrum_strip(self, figsize=(5, 1.3), equalize_sd_amplitude=True, overlay_graph=True):
        if len(self.spectral_distribution) == 0:
            return b''

        plotdata = io.BytesIO()
        fig, ax = plt.subplots(figsize=figsize)
        if overlay_graph:
            line = ax.twinx()
            line.plot(self.stardata['WAVE']/10.0, self.stardata['FLUX_CORR'],
                      color='white', zorder=-1, linewidth=0.5)
        # Settings from https://github.com/colour-science/colour/blob/master/colour/plotting/common.py
        try:
            cplt.plot_single_sd(self.spectral_distribution, modulate_colours_with_sd_amplitude=True, equalize_sd_amplitude=equalize_sd_amplitude,
                                y_label='', x_label='', title='', y_ticker=False, figure=fig, axes=ax, filename=plotdata)
        except Exception:
            line.plot(self.stardata['WAVE']/10.0, self.stardata['FLUX_CORR'],
                      color='black', zorder=0, linewidth=0.5)
            plt.savefig(plotdata, format='png')

        plt.close()
        return plotdata.getvalue()

    def show_spectrum_strip(self, figsize=(10, 2), equalize_sd_amplitude=False):
        display(Image(self.get_spectrum_strip(figsize)))

    def get_simulated_star_image(self, figsize=(1, 1)):
        plotdata = io.BytesIO()
        star = plt.Circle(
            (0.5, 0.5), 0.15, color=self.rgb)
        cosmos = plt.Rectangle((0.0, 0.0), 1.0, 1.0, color=[0, 0, 0])

        _, ax = plt.subplots(figsize=figsize)
        ax.set_axis_off()
        ax.add_patch(cosmos)
        ax.add_patch(star)
        plt.savefig(plotdata, format='png')
        plt.close()

        return plotdata.getvalue()

    def show_simulated_star_image(self, figsize=(2, 2)):
        display(Image(self.get_simulated_star_image(figsize)))

    def get_table_line(self):

        o = '<tr>'
        o += '<td>%s</td>' % (self.get_mangaid())

        starimg = self.get_star_image()
        if len(starimg) == 0:
            starimg = self.get_simulated_star_image()
        starimg_enc = base64.b64encode(starimg).decode('ascii')
        o += '<td><img src="data:image/png;base64,%s"></td>' % starimg_enc

        sp = self.get_spectrum_strip()
        if len(sp) > 0:
            starspec_enc = base64.b64encode(sp).decode('ascii')
            o += '<td><img src="data:image/png;base64,%s"></td>' % starspec_enc
        else:
            o += '<td>No strip</td>'

        o += "<td>%.0fK</td><td>%s</td>" % (self.stardata['INPUT_TEFF'],
                                            self.stardata['INPUT_SOURCE'])
        o += "</tr>\n"

        return o
