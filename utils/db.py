"""Gestión del dataset y del historial de predicciones en SQLite (PASO 1 de la ruta)."""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import RUTA_DB

ESQUEMA = """
CREATE TABLE IF NOT EXISTS imagenes (
    id            INTEGER PRIMARY KEY,
    archivo       TEXT NOT NULL UNIQUE,      -- ruta relativa a data/raw/Dataset_BUSI_with_GT
    clase         TEXT NOT NULL CHECK (clase IN ('normal','benigno','maligno')),
    ancho         INTEGER, alto INTEGER, canales INTEGER,
    intensidad_media REAL,
    n_mascaras    INTEGER,                   -- nº de máscaras de segmentación asociadas
    area_lesion   REAL,                      -- fracción de la imagen ocupada por la lesión
    md5           TEXT,
    phash         TEXT,
    grupo_dup     INTEGER,                   -- imágenes casi idénticas comparten grupo
    excluida      INTEGER DEFAULT 0,         -- 1 = descartada en la depuración
    motivo        TEXT,                      -- motivo de la exclusión
    particion     TEXT CHECK (particion IN ('train','val','test') OR particion IS NULL)
);
CREATE TABLE IF NOT EXISTS predicciones (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    fecha         TEXT NOT NULL,
    origen        TEXT,                      -- nombre del archivo subido o del ejemplo
    modelo        TEXT,
    clase_predicha TEXT,
    p_normal      REAL, p_benigno REAL, p_maligno REAL
);
CREATE INDEX IF NOT EXISTS ix_img_clase ON imagenes(clase);
CREATE INDEX IF NOT EXISTS ix_img_part  ON imagenes(particion);
"""


def conectar(ruta: Path | str = RUTA_DB) -> sqlite3.Connection:
    con = sqlite3.connect(ruta, check_same_thread=False)
    con.executescript(ESQUEMA)
    return con


def cargar_imagenes(con: sqlite3.Connection, df: pd.DataFrame) -> int:
    """Inserta (o reemplaza) el inventario de imágenes."""
    cols = ["archivo", "clase", "ancho", "alto", "canales", "intensidad_media", "n_mascaras",
            "area_lesion", "md5", "phash", "grupo_dup", "excluida", "motivo", "particion"]
    filas = df.reindex(columns=cols).astype(object).where(df.reindex(columns=cols).notna(), None)
    con.execute("DELETE FROM imagenes")
    con.executemany(f"INSERT INTO imagenes ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                    filas.itertuples(index=False, name=None))
    con.commit()
    return con.execute("SELECT COUNT(*) FROM imagenes").fetchone()[0]


def asignar_particion(con: sqlite3.Connection, particiones: dict[str, str]) -> None:
    con.executemany("UPDATE imagenes SET particion = ? WHERE archivo = ?",
                    [(p, a) for a, p in particiones.items()])
    con.commit()


def consultar(con: sqlite3.Connection, sql: str, params: tuple = ()) -> pd.DataFrame:
    return pd.read_sql_query(sql, con, params=params)


def registrar_prediccion(con: sqlite3.Connection, origen: str, modelo: str, clase: str, probs) -> None:
    con.execute("INSERT INTO predicciones (fecha, origen, modelo, clase_predicha, p_normal, p_benigno, p_maligno) "
                "VALUES (?,?,?,?,?,?,?)",
                (datetime.now().isoformat(timespec="seconds"), origen, modelo, clase, *map(float, probs)))
    con.commit()
