--
-- PostgreSQL database dump
--

-- Dumped from database version 17.5
-- Dumped by pg_dump version 17.5

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
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
-- Name: found_items; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.found_items (
    id integer NOT NULL,
    image_path text NOT NULL,
    description text NOT NULL,
    location text,
    found_at timestamp without time zone,
    reporter text
);


--
-- Name: found_items_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.found_items_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: found_items_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.found_items_id_seq OWNED BY public.found_items.id;


--
-- Name: found_items id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.found_items ALTER COLUMN id SET DEFAULT nextval('public.found_items_id_seq'::regclass);


--
-- Data for Name: found_items; Type: TABLE DATA; Schema: public; Owner: -
--

INSERT INTO public.found_items VALUES (1, 'data/reported/images/tas_hitam_fun_boy_aula_f.jpg', 'Tas ransel hitam polos merk funboy, ditemukan di Aula Gedung F', 'Aula Gedung F', '2025-12-04 21:04:23.824584', 'Penemu Testing');
INSERT INTO public.found_items VALUES (2, 'data/reported/images/tas_hitam_fun_boy_aula_f.jpg', 'Tas ransel hitam polos merk funboy, ditemukan di Aula Gedung F', 'Aula Gedung F', '2025-12-04 21:32:07.288113', 'Alief');


--
-- Name: found_items_id_seq; Type: SEQUENCE SET; Schema: public; Owner: -
--

SELECT pg_catalog.setval('public.found_items_id_seq', 2, true);


--
-- Name: found_items found_items_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.found_items
    ADD CONSTRAINT found_items_pkey PRIMARY KEY (id);


--
-- Name: idx_found_items_description_gin; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_found_items_description_gin ON public.found_items USING gin (to_tsvector('simple'::regconfig, description));


--
-- Name: idx_found_items_location; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_found_items_location ON public.found_items USING btree (location);


--
-- PostgreSQL database dump complete
--

