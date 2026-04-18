--
-- PostgreSQL database dump
--

\restrict AdyLzmQYHb8T2gk2M5fohqnbQS0NUkVr0E0zbkNQM8tTUrgEuXIrLbUhJ15JzN8

-- Dumped from database version 15.17
-- Dumped by pg_dump version 15.17

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: trades; Type: TABLE; Schema: public; Owner: trader
--

CREATE TABLE public.trades (
    id integer NOT NULL,
    ticker character varying(10),
    action character varying(10),
    quantity numeric,
    quoted_price numeric,
    fill_price numeric,
    status character varying(20),
    "timestamp" timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.trades OWNER TO trader;

--
-- Name: trades_id_seq; Type: SEQUENCE; Schema: public; Owner: trader
--

CREATE SEQUENCE public.trades_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.trades_id_seq OWNER TO trader;

--
-- Name: trades_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: trader
--

ALTER SEQUENCE public.trades_id_seq OWNED BY public.trades.id;


--
-- Name: trades id; Type: DEFAULT; Schema: public; Owner: trader
--

ALTER TABLE ONLY public.trades ALTER COLUMN id SET DEFAULT nextval('public.trades_id_seq'::regclass);


--
-- Data for Name: trades; Type: TABLE DATA; Schema: public; Owner: trader
--

COPY public.trades (id, ticker, action, quantity, quoted_price, fill_price, status, "timestamp") FROM stdin;
1	BTC	BUY	0.00308	65000.0	64957.93	FILLED	2026-04-03 05:20:57.191731
\.


--
-- Name: trades_id_seq; Type: SEQUENCE SET; Schema: public; Owner: trader
--

SELECT pg_catalog.setval('public.trades_id_seq', 1, true);


--
-- Name: trades trades_pkey; Type: CONSTRAINT; Schema: public; Owner: trader
--

ALTER TABLE ONLY public.trades
    ADD CONSTRAINT trades_pkey PRIMARY KEY (id);


--
-- PostgreSQL database dump complete
--

\unrestrict AdyLzmQYHb8T2gk2M5fohqnbQS0NUkVr0E0zbkNQM8tTUrgEuXIrLbUhJ15JzN8

