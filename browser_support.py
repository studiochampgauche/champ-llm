"""Select Playwright engines and Google Chrome without silent browser fallbacks."""
import os

BROWSER_CHOICES = ('chromium', 'chrome', 'firefox', 'webkit', 'safari')
BROWSER_LABELS = {
    'chromium': 'Chromium',
    'chrome': 'Google Chrome',
    'firefox': 'Firefox (Playwright build)',
    'webkit': 'WebKit (Safari engine, not the Safari application)',
}


def normalize_browser(browser):
    if not isinstance(browser, str) or browser not in BROWSER_CHOICES:
        raise ValueError('Choose a browser from: ' + ', '.join(BROWSER_CHOICES))
    return 'webkit' if browser == 'safari' else browser


def normalize_browsers(browsers=None):
    if browsers is None:
        return ('chromium',)
    if isinstance(browsers, str):
        browsers = [browsers]
    selected = tuple(dict.fromkeys(normalize_browser(name) for name in browsers))
    if not selected:
        raise ValueError('Select at least one browser')
    return selected


def browser_info(browser):
    browser = normalize_browser(browser)
    return {'browser': browser, 'browser_label': BROWSER_LABELS[browser],
            'engine': 'chromium' if browser == 'chrome' else browser,
            'channel': 'chrome' if browser == 'chrome' else None}


def launch_browser(playwright, browser):
    info = browser_info(browser)
    options = {'headless': True}
    if info['browser'] == 'chrome':
        options['channel'] = 'chrome'
    elif info['browser'] == 'chromium':
        executable = (os.environ.get('CHAMP_LLM_CHROMIUM') or
                      os.environ.get('CHAMP_GAUCHE_LLM_CHROMIUM') or
                      os.environ.get('MONCERVEAU_CHROMIUM'))
        if executable:
            options.update(executable_path=executable, args=['--no-sandbox'])
    try:
        return getattr(playwright, info['engine']).launch(**options)
    except Exception as exc:
        install = ('Install Google Chrome on this machine, or explicitly select --browser chromium.'
                   if info['browser'] == 'chrome' else
                   f"Install the matching browser: python -m playwright install {info['engine']}. "
                   'On supported Linux systems, --with-deps also installs system libraries.')
        raise RuntimeError(f"Cannot launch {info['browser_label']}. {install} "
                           f'No fallback was used. Original error: {exc}') from exc


def add_browser_argument(parser):
    parser.add_argument('--browser', action='append', choices=BROWSER_CHOICES,
                        help='Browser to inspect (default: chromium). Repeat for several browsers. '
                             'chrome uses installed Google Chrome; safari is an alias for WebKit, not the Safari app.')
