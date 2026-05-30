"""Extraction contract for Ministerio da Saude arbovirus bulletins.

Every field here captures signal that is NOT already in the SINAN/SIVEP
case-count series. The test each field must pass: does it carry information
*beyond* the contemporaneous case number? If not, it risks leaking the
forecasting target into the feature matrix.

The bulletins routinely restate the very counts we predict, so the value
of this modality is the qualitative, forward-looking intel around those
counts -- not the counts themselves.
"""

from datetime import date, datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator

# Bump this constant on any field addition, removal, or type change.
# The version is stored in every output record so rows can always be
# traced back to the schema that produced them.
SCHEMA_VERSION = "1.1.0"

# Audit cap on the verbatim evidence quote. Declared as a constant because it
# is enforced in two places that must agree: the Field(max_length=...) contract
# and the repairing validator below.
EVIDENCE_SPAN_MAX_CHARS = 280


# --- Enums -------------------------------------------------------------------
# Each ordinal enum includes a "not stated" member so the model has a valid
# abstain path when the bulletin is silent. This is the primary mechanism
# for suppressing hallucination: the model never needs to invent a value.

class AlertLevel(str, Enum):
    """Official alert or risk level the bulletin assigns to the region."""
    none = "sem_alerta"
    attention = "atencao"
    alert = "alerta"
    emergency = "emergencia"
    not_stated = "nao_informado"


class ReportedTrend(str, Enum):
    """Narrative trend the bulletin DESCRIBES in prose (not the raw count)."""
    falling = "queda"
    stable = "estavel"
    rising = "alta"
    not_stated = "nao_informado"


class ActionIntensity(str, Enum):
    """Intensity of the control actions the bulletin recommends."""
    none = "nenhuma"
    routine = "rotina"
    intensified = "intensificacao"
    emergency = "emergencial"
    not_stated = "nao_informado"


class Serotype(str, Enum):
    """Dengue serotypes the bulletin names as currently circulating."""
    denv1 = "DENV-1"
    denv2 = "DENV-2"
    denv3 = "DENV-3"
    denv4 = "DENV-4"


class GeographicScope(str, Enum):
    """Whether the passage covers Brazil as a whole or one specific state.

    `not_applicable` is the abstain path for bulletins that are not about
    arboviroses at all (the corpus is fed in unfiltered, so most bulletins are
    off-topic). Without it the model would be forced to invent a scope on, say,
    an antimicrobial-resistance bulletin.
    """
    national = "nacional"
    state = "estadual"
    not_applicable = "nao_se_aplica"


# --- Extraction model --------------------------------------------------------

class BulletinSignal(BaseModel):
    """The qualitative signal the model extracts from one bulletin passage.

    Field descriptions double as extraction instructions because Instructor
    passes this schema to the model. They are written in Portuguese to match
    the source corpus and reduce translation ambiguity.
    """

    # Primary classification output. The corpus is fed in unfiltered (every
    # Ministerio da Saude bulletin, not just arbovirus ones), so deciding
    # relevance IS the experiment -- the parser never pre-filters. When this is
    # False, every field below should take its abstain value (nao_se_aplica /
    # nao_informado / empty list).
    is_arbovirus_related: bool = Field(
        description=(
            "O boletim trata de arboviroses (dengue, chikungunya, Zika, febre amarela "
            "ou similares)? Muitos boletins do Ministerio da Saude tratam de outros "
            "temas (resistencia antimicrobiana, raiva, sarampo, etc.) e NAO sao "
            "relevantes: marque false e use 'nao_se_aplica'/'nao_informado' nos demais "
            "campos."
        ),
    )
    geographic_scope: GeographicScope = Field(
        description="O trecho fala do Brasil como um todo (nacional) ou de uma UF especifica (estadual)?"
    )
    uf: Optional[str] = Field(
        default=None,
        description="Sigla da UF (ex.: 'BA', 'SP') quando o escopo for estadual; null se nacional.",
        max_length=2,
    )
    reported_trend: ReportedTrend = Field(
        description="Tendencia que o texto NARRA (alta/queda/estavel), nao o numero de casos."
    )
    alert_level: AlertLevel = Field(
        description="Nivel de alerta/risco que o boletim atribui a regiao."
    )
    serotypes_mentioned: list[Serotype] = Field(
        default_factory=list,
        description="Sorotipos de dengue citados como circulantes nesta regiao.",
    )
    emerging_or_new_serotype: bool = Field(
        default=False,
        description=(
            "O texto sinaliza um sorotipo novo ou reemergente na regiao? "
            "(intel qualitativa, antecede mudancas de incidencia)"
        ),
    )
    forward_looking_warning: bool = Field(
        default=False,
        description=(
            "Ha alerta explicito sobre as PROXIMAS semanas ou a temporada que vem? "
            "(o sinal mais limpo, pois nao reflete a contagem atual)"
        ),
    )
    regions_of_concern: list[str] = Field(
        default_factory=list,
        description="Municipios ou regioes destacados como preocupantes.",
    )
    recommended_action_intensity: ActionIntensity = Field(
        description="Intensidade das acoes recomendadas pelo boletim."
    )
    evidence_span: str = Field(
        description=(
            "Trecho verbatim curto do boletim que justifica a avaliacao acima. "
            "Para auditoria humana."
        ),
        max_length=EVIDENCE_SPAN_MAX_CHARS,
    )
    requires_human_review: bool = Field(
        default=False,
        description="True quando o trecho e ambiguo ou voce nao tem confianca na extracao.",
    )

    @model_validator(mode="before")
    @classmethod
    def _clip_overlong_evidence(cls, data: object) -> object:
        """Clip an over-long evidence_span to the audit cap and flag it for review.

        Constrained decoding guarantees structure and enum membership, but no
        JSON-schema decoder (XGrammar, OpenAI, or Gemini structured output)
        enforces string max_length -- the cap is dropped when the schema becomes
        a decoder grammar. So the model can legitimately emit an evidence_span
        longer than the cap, which would otherwise fail validation and, after
        instructor's retries, abort the entire batch on one passage.

        A clipped verbatim prefix is still auditable; a crashed run is not. We
        therefore truncate to the cap and force requires_human_review=True so the
        over-run surfaces in the review queue instead of being silently trusted.
        """
        if not isinstance(data, dict):
            return data

        span = data.get("evidence_span")
        if isinstance(span, str) and len(span) > EVIDENCE_SPAN_MAX_CHARS:
            # rstrip avoids leaving a dangling partial word's trailing space.
            data["evidence_span"] = span[:EVIDENCE_SPAN_MAX_CHARS].rstrip()
            data["requires_human_review"] = True

        return data


# --- Provenance wrapper ------------------------------------------------------

class ExtractionRecord(BaseModel):
    """One archived row: the model's signal plus the provenance the code stamps.

    Provenance fields (model_id, prompt_version, schema_version, extracted_at)
    are never produced by the LLM -- they are added in pipeline.py so every
    row is traceable to an exact model, prompt, and schema version.
    """

    source_file: str
    # Best-effort publication date parsed from the bulletin cover. Optional
    # because the corpus is unfiltered and some covers do not yield a parseable
    # date; a missing date must not abort the batch (it flags the row instead).
    bulletin_publication_date: Optional[date] = None
    # The epidemiological week the bulletin reports ON (not its publication date).
    # Bulletins publish with a lag; the forecasting model must only see features
    # from weeks strictly before the prediction horizon. Optional and left unset
    # by the parser: most bulletins carry no SE, and inferring the *reported*
    # week from a multi-week report is a downstream alignment concern, not the
    # parser's. Stored as `YYYY-SENN` when known.
    epi_week_reported: Optional[str] = None
    signal: BulletinSignal

    model_id: str
    prompt_version: str
    schema_version: str = SCHEMA_VERSION
    extracted_at: datetime
