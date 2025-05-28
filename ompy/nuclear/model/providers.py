from ..readers.ripl3.reader import RIPL3Record, GammaRecord
from .levels import GammaBranch

@register_provider('from_ripl3_gamma_record', 
                   from=GammaRecord, 
                   to=GammaBranch)
def gamma_record_provider(record: GammaRecord) -> GammaBranch:
    return GammaBranch(final=record.Nf,
                       Eg=record.Eg,
                       Pg=record.Pg,
                       Pem=record.Pe,
                       ICC=record.ICC)

