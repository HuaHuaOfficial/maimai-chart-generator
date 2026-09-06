from .simai import parse_inote_ticks

def fixture_events(fixture):
    explicit=fixture.get('eventWindow')
    if explicit is not None:return [(int(x['tick']),str(x['text'])) for x in explicit]
    return list(parse_inote_ticks(fixture['inote'],float(fixture['bpm'])).events)
