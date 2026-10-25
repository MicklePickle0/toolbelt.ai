-- Event DB: the source of truth. Stats are SQL queries over these tables,
-- never recomputed from video.
-- Written to run on SQLite; keep it portable to DuckDB where practical.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS teams (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS players (
    id            INTEGER PRIMARY KEY,
    team_id       INTEGER NOT NULL REFERENCES teams(id),
    jersey_number INTEGER NOT NULL,
    name          TEXT NOT NULL,
    is_libero     INTEGER NOT NULL DEFAULT 0,
    UNIQUE (team_id, jersey_number)
);

CREATE TABLE IF NOT EXISTS games (
    id            INTEGER PRIMARY KEY,
    team_id       INTEGER NOT NULL REFERENCES teams(id),
    video_key     TEXT NOT NULL UNIQUE,        -- stable id from the manifest
    opponent      TEXT,
    played_on     TEXT,                       -- ISO date
    venue         TEXT NOT NULL,              -- gym / tournament site; splits are grouped by it
    -- Tape height for this game. Everything that uses it (calibration scale,
    -- block check, ball height) reads this column. Preset names are in
    -- config/court.yaml; NULL preset means a custom height.
    net_height_m      REAL NOT NULL CHECK (net_height_m > 0),
    net_height_preset TEXT,
    ball          TEXT NOT NULL,              -- key of `balls` in config/court.yaml
    video_path    TEXT NOT NULL,
    fps           REAL NOT NULL,
    frame_count   INTEGER,
    calibration   TEXT,                       -- JSON: camera pose, intrinsics, confidence
    created_at    TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sets (
    id               INTEGER PRIMARY KEY,
    game_id          INTEGER NOT NULL REFERENCES games(id),
    set_number       INTEGER NOT NULL,
    starting_lineup  TEXT,                    -- JSON: six jersey numbers in rotation order
    start_frame      INTEGER,
    end_frame        INTEGER,
    UNIQUE (game_id, set_number)
);

CREATE TABLE IF NOT EXISTS rallies (
    id                  INTEGER PRIMARY KEY,
    set_id              INTEGER NOT NULL REFERENCES sets(id),
    rally_number        INTEGER NOT NULL,
    start_frame         INTEGER NOT NULL,
    end_frame           INTEGER NOT NULL,
    start_t             REAL NOT NULL,
    end_t               REAL NOT NULL,
    segment_confidence  REAL,
    serving_team        TEXT CHECK (serving_team IN ('us', 'them')),
    rotation            INTEGER CHECK (rotation BETWEEN 1 AND 6),
    -- Point winner comes from the review UI key press (ground truth) or
    -- scoreboard OCR as a sanity check; never from the ML pipeline alone.
    point_winner        TEXT CHECK (point_winner IN ('us', 'them')),
    point_winner_source TEXT CHECK (point_winner_source IN ('user', 'ocr')),
    score_us            INTEGER,
    score_them          INTEGER,
    outcome             TEXT,                 -- kill, attack_error, reception_error, ace, service_error, ...
    outcome_confidence  REAL,
    reviewed            INTEGER NOT NULL DEFAULT 0,
    UNIQUE (set_id, rally_number)
);

CREATE TABLE IF NOT EXISTS tracks (
    id                  INTEGER PRIMARY KEY,
    game_id             INTEGER NOT NULL REFERENCES games(id),
    tracker_track_id    INTEGER NOT NULL,     -- id assigned by the tracker
    first_frame         INTEGER NOT NULL,
    last_frame          INTEGER NOT NULL,
    player_id           INTEGER REFERENCES players(id),
    player_confidence   REAL,
    number_votes        TEXT,                 -- JSON: {jersey_number: weight}
    reviewed            INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS contacts (
    id                  INTEGER PRIMARY KEY,
    rally_id            INTEGER NOT NULL REFERENCES rallies(id),
    t                   REAL NOT NULL,
    frame               INTEGER NOT NULL,
    contact_confidence  REAL,
    team                TEXT CHECK (team IN ('us', 'them')),
    player_id           INTEGER REFERENCES players(id),
    track_id            INTEGER REFERENCES tracks(id),
    action              TEXT,                 -- serve, receive, set, attack, block, dig, dump, free_ball, coverage, ...
    action_confidence   REAL,
    ball_x              REAL,
    ball_y              REAL,
    ball_z              REAL,
    outcome             TEXT,
    pass_score          INTEGER CHECK (pass_score BETWEEN 0 AND 4),
    pass_confidence     REAL,
    reviewed            INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS ball_samples (
    id          INTEGER PRIMARY KEY,
    game_id     INTEGER NOT NULL REFERENCES games(id),
    rally_id    INTEGER REFERENCES rallies(id),
    frame       INTEGER NOT NULL,
    t           REAL NOT NULL,
    u           REAL,                         -- image px
    v           REAL,
    x           REAL,                         -- court metres (see config/court.yaml)
    y           REAL,
    z           REAL,
    confidence  REAL,
    source      TEXT NOT NULL CHECK (source IN ('detection', 'kalman', 'arc_fit', 'label'))
);

-- Every human correction is kept: it is both an audit log and training data.
CREATE TABLE IF NOT EXISTS corrections (
    id          INTEGER PRIMARY KEY,
    table_name  TEXT NOT NULL,
    row_id      INTEGER NOT NULL,
    field       TEXT NOT NULL,
    old_value   TEXT,
    new_value   TEXT,
    propagated_from INTEGER REFERENCES corrections(id),  -- set when applied across a tracklet
    author      TEXT,
    created_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_games_venue ON games(venue);
CREATE INDEX IF NOT EXISTS idx_contacts_rally ON contacts(rally_id);
CREATE INDEX IF NOT EXISTS idx_contacts_player ON contacts(player_id);
CREATE INDEX IF NOT EXISTS idx_ball_samples_game_frame ON ball_samples(game_id, frame);
CREATE INDEX IF NOT EXISTS idx_tracks_game ON tracks(game_id);
CREATE INDEX IF NOT EXISTS idx_rallies_set ON rallies(set_id);
