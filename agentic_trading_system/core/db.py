import psycopg2.extensions


def ensure_trades_table(conn: psycopg2.extensions.connection) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                id SERIAL PRIMARY KEY,
                order_id VARCHAR(64),
                proposal_id VARCHAR(64),
                decision_id VARCHAR(64),
                ticker VARCHAR(10),
                action VARCHAR(10),
                quantity NUMERIC,
                quoted_price NUMERIC,
                fill_price NUMERIC,
                slippage_bps NUMERIC,
                status VARCHAR(20),
                paper_trade BOOLEAN DEFAULT TRUE,
                reasoning TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        cur.execute("ALTER TABLE trades ADD COLUMN IF NOT EXISTS order_id VARCHAR(64)")
        cur.execute("ALTER TABLE trades ADD COLUMN IF NOT EXISTS proposal_id VARCHAR(64)")
        cur.execute("ALTER TABLE trades ADD COLUMN IF NOT EXISTS decision_id VARCHAR(64)")
        cur.execute("ALTER TABLE trades ADD COLUMN IF NOT EXISTS ticker VARCHAR(10)")
        cur.execute("ALTER TABLE trades ADD COLUMN IF NOT EXISTS action VARCHAR(10)")
        cur.execute("ALTER TABLE trades ADD COLUMN IF NOT EXISTS quantity NUMERIC")
        cur.execute("ALTER TABLE trades ADD COLUMN IF NOT EXISTS quoted_price NUMERIC")
        cur.execute("ALTER TABLE trades ADD COLUMN IF NOT EXISTS fill_price NUMERIC")
        cur.execute("ALTER TABLE trades ADD COLUMN IF NOT EXISTS slippage_bps NUMERIC")
        cur.execute("ALTER TABLE trades ADD COLUMN IF NOT EXISTS status VARCHAR(20)")
        cur.execute(
            "ALTER TABLE trades ADD COLUMN IF NOT EXISTS paper_trade BOOLEAN DEFAULT TRUE"
        )
        cur.execute("ALTER TABLE trades ADD COLUMN IF NOT EXISTS reasoning TEXT")
        cur.execute(
            """
            ALTER TABLE trades
            ADD COLUMN IF NOT EXISTS timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            """
        )
        cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS trades_order_id_idx ON trades(order_id)")
    conn.commit()
