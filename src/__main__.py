"""Entry point for running the extraction pipeline from the command line.

Usage:
    python src/ --input data/passages/passages.jsonl \
                --output data/extracted/signals.jsonl

    # Or with a synthetic demo passage (no --input flag needed):
    python src/

The demo mode runs a single hard-coded passage so you can verify the full
pipeline is wired correctly before you have real PDF passages to process.

Everything that affects what gets extracted (model, sampling, prompt, paths) is
controlled by config.yaml; connection secrets come from .env (see .env.example).
CLI flags override the config data paths for ad-hoc runs. .env is loaded by
config.py on import, so no explicit load_dotenv() is needed here.
"""

import argparse
import json
from pathlib import Path

from config import get_settings
from pipeline import run_corpus


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract qualitative signals from dengue bulletin passages."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="JSONL file produced by parse.py. Omit to run the built-in demo passage.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSONL file where ExtractionRecords are appended (default: data.signals from config.yaml).",
    )
    return parser.parse_args()


# A synthetic, full-length bulletin used when no --input file is provided. It is
# deliberately long and mixed: real arbovirus signal (DENV-3 reemergente na Bahia,
# nivel de alerta, municipios de preocupacao, aviso para as proximas semanas) is
# buried among administrative boilerplate, methodology, and an unrelated disease
# section. This lets the optional preprocessing stage visibly filter the noise and
# gives both models real text to reason over.
_DEMO_PASSAGES = [
    {
        "source_file": "boletim_se18_2024.pdf",
        "bulletin_publication_date": "2024-05-10",
        "epi_week_reported": "2024-SE18",
        "text": (
            "MINISTERIO DA SAUDE\n"
            "Secretaria de Vigilancia em Saude e Ambiente\n"
            "Boletim Epidemiologico - Volume 55, No 9 - Maio de 2024\n"
            "\n"
            "1. APRESENTACAO\n"
            "Este boletim consolida informacoes das Secretarias Estaduais de Saude "
            "referentes a Semana Epidemiologica 18 de 2024 (28/04 a 04/05). Os dados "
            "sao preliminares e estao sujeitos a revisao apos consolidacao no Sistema "
            "de Informacao de Agravos de Notificacao. A reproducao e permitida desde "
            "que citada a fonte. Demandas a Ouvidoria pelo telefone 136.\n"
            "\n"
            "2. METODOLOGIA\n"
            "As notificacoes foram extraidas em 06/05/2024. Casos provaveis excluem os "
            "descartados por criterio laboratorial. A classificacao por unidade "
            "federativa segue o municipio de residencia. Series historicas utilizam "
            "media movel de quatro semanas para suavizar atrasos de digitacao.\n"
            "\n"
            "3. SITUACAO DAS ARBOVIROSES\n"
            "Na Bahia, observa-se tendencia de alta sustentada no numero de casos "
            "provaveis de dengue nas ultimas seis semanas, com interiorizacao da "
            "transmissao. Chama atencao a circulacao predominante do sorotipo DENV-3, "
            "reintroduzido no estado apos anos de baixa circulacao, o que amplia a "
            "populacao suscetivel e o risco de formas graves. As regioes de saude de "
            "Feira de Santana, Vitoria da Conquista e Barreiras concentram a maior "
            "parte das notificacoes e foram classificadas em nivel de ALERTA pela "
            "vigilancia estadual.\n"
            "A Secretaria recomenda a intensificacao imediata das acoes de controle "
            "vetorial e a preparacao das unidades de saude. Alerta-se que, com a "
            "persistencia das chuvas e das temperaturas elevadas, espera-se aumento "
            "adicional da incidencia nas proximas semanas, exigindo monitoramento "
            "reforcado ao longo de toda a temporada 2024.\n"
            "Nos demais estados do Nordeste a situacao permanece estavel, sem "
            "alteracao do padrao sazonal esperado.\n"
            "\n"
            "4. SARAMPO\n"
            "Nao foram confirmados casos autoctones de sarampo no periodo. O pais "
            "mantem o monitoramento de casos suspeitos e reforca a importancia da "
            "cobertura vacinal homogenea com a triplice viral. Esta secao nao guarda "
            "relacao com as arboviroses e e apresentada apenas para registro.\n"
            "\n"
            "5. CONSIDERACOES FINAIS\n"
            "A vigilancia continua sera atualizada semanalmente. Equipes municipais "
            "devem reportar inconsistencias a referencia estadual."
        ),
    }
]


def main() -> None:
    args = _parse_args()
    settings = get_settings()

    if args.input is not None:
        with args.input.open(encoding="utf-8") as fh:
            passages = [json.loads(line) for line in fh if line.strip()]
    else:
        print("No --input provided. Running demo passage.")
        passages = _DEMO_PASSAGES

    output_path = args.output or settings.data.signals
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_corpus(passages, output_path)
    print(f"Records written to {output_path.resolve()}")


if __name__ == "__main__":
    main()
