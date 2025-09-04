-- phpMyAdmin SQL Dump
-- version 2.11.6
-- http://www.phpmyadmin.net
--
-- Host: localhost
-- Generation Time: Dec 06, 2024 at 05:12 AM
-- Server version: 5.0.51
-- PHP Version: 5.2.6

SET SQL_MODE="NO_AUTO_VALUE_ON_ZERO";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;

--
-- Database: `criminal_face`
--

-- --------------------------------------------------------

--
-- Table structure for table `cf_admin`
--

CREATE TABLE `cf_admin` (
  `username` varchar(20) NOT NULL,
  `password` varchar(20) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `cf_admin`
--

INSERT INTO `cf_admin` (`username`, `password`) VALUES
('admin', 'admin');

-- --------------------------------------------------------

--
-- Table structure for table `cf_camera`
--

CREATE TABLE `cf_camera` (
  `id` int(11) NOT NULL,
  `camera` varchar(20) NOT NULL,
  `station_id` varchar(20) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `cf_camera`
--

INSERT INTO `cf_camera` (`id`, `camera`, `station_id`) VALUES
(1, 'Camera12', 'S001'),
(2, 'Camera17', 'S001');

-- --------------------------------------------------------

--
-- Table structure for table `cf_criminal_alert`
--

CREATE TABLE `cf_criminal_alert` (
  `id` int(11) NOT NULL,
  `cid` int(11) NOT NULL,
  `name` varchar(20) NOT NULL,
  `face_image` varchar(20) NOT NULL,
  `camera` varchar(20) NOT NULL,
  `area` varchar(30) NOT NULL,
  `city` varchar(20) NOT NULL,
  `station_id` varchar(20) NOT NULL,
  `create_date` timestamp NOT NULL default CURRENT_TIMESTAMP on update CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `cf_criminal_alert`
--


-- --------------------------------------------------------

--
-- Table structure for table `cf_criminal_details`
--

CREATE TABLE `cf_criminal_details` (
  `id` int(11) NOT NULL,
  `name` varchar(20) NOT NULL,
  `gender` varchar(10) NOT NULL,
  `dob` varchar(20) NOT NULL,
  `address` varchar(50) NOT NULL,
  `complaint` varchar(100) NOT NULL,
  `complainant_name` varchar(20) NOT NULL,
  `complainant_address` varchar(50) NOT NULL,
  `place` varchar(50) NOT NULL,
  `complaint_date` varchar(20) NOT NULL,
  `district` varchar(20) NOT NULL,
  `fir_date` varchar(20) NOT NULL,
  `jail_period` varchar(50) NOT NULL,
  `release_date` varchar(20) NOT NULL,
  `proof` varchar(50) NOT NULL,
  `police_station` varchar(20) NOT NULL,
  `police_inspector` varchar(20) NOT NULL,
  `entryby` varchar(20) NOT NULL,
  `register_date` varchar(20) NOT NULL,
  `fimg` varchar(20) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `cf_criminal_details`
--


-- --------------------------------------------------------

--
-- Table structure for table `cf_police`
--

CREATE TABLE `cf_police` (
  `id` int(11) NOT NULL,
  `station_id` varchar(20) NOT NULL,
  `police_name` varchar(20) NOT NULL,
  `mobile` bigint(20) NOT NULL,
  `email` varchar(40) NOT NULL,
  `police_id` varchar(20) NOT NULL,
  `designation` varchar(20) NOT NULL,
  `create_date` timestamp NOT NULL default CURRENT_TIMESTAMP on update CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `cf_police`
--

INSERT INTO `cf_police` (`id`, `station_id`, `police_name`, `mobile`, `email`, `police_id`, `designation`, `create_date`) VALUES
(1, 'S001', 'Sankar', 9845484226, 'sankar@gmail.com', 'P001', 'Inspector', '2024-02-10 10:03:04');

-- --------------------------------------------------------

--
-- Table structure for table `cf_policestation`
--

CREATE TABLE `cf_policestation` (
  `id` int(11) NOT NULL,
  `station_name` varchar(20) NOT NULL,
  `mobile` bigint(20) NOT NULL,
  `email` varchar(40) NOT NULL,
  `area` varchar(30) NOT NULL,
  `city` varchar(30) NOT NULL,
  `station_id` varchar(20) NOT NULL,
  `password` varchar(20) NOT NULL,
  `create_date` timestamp NOT NULL default CURRENT_TIMESTAMP on update CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `cf_policestation`
--

INSERT INTO `cf_policestation` (`id`, `station_name`, `mobile`, `email`, `area`, `city`, `station_id`, `password`, `create_date`) VALUES
(1, 'B1 Station', 9894442716, 'b1station@gmail.com', 'BS Nagar', 'Chennai', 'S001', '123456', '2024-02-11 10:32:50');

-- --------------------------------------------------------

--
-- Table structure for table `cf_video`
--

CREATE TABLE `cf_video` (
  `id` int(11) NOT NULL,
  `filename` varchar(20) NOT NULL,
  `date_time` timestamp NOT NULL default CURRENT_TIMESTAMP on update CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `cf_video`
--

INSERT INTO `cf_video` (`id`, `filename`, `date_time`) VALUES
(18, 'v18.avi', '2024-02-11 22:59:18'),
(19, 'v19.avi', '2024-02-11 23:00:29');

-- --------------------------------------------------------

--
-- Table structure for table `vt_face`
--

CREATE TABLE `vt_face` (
  `id` int(11) NOT NULL,
  `vid` int(11) NOT NULL,
  `vface` varchar(30) NOT NULL,
  `mask_st` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `vt_face`
--

INSERT INTO `vt_face` (`id`, `vid`, `vface`, `mask_st`) VALUES
(27, 2, 'User.2.2.jpg', 0),
(28, 2, 'User.2.3.jpg', 0),
(29, 2, 'User.2.4.jpg', 0),
(30, 2, 'User.2.5.jpg', 0),
(31, 2, 'User.2.6.jpg', 0),
(32, 2, 'User.2.7.jpg', 0),
(33, 2, 'User.2.8.jpg', 0),
(34, 2, 'User.2.9.jpg', 0),
(35, 2, 'User.2.10.jpg', 0),
(36, 2, 'User.2.11.jpg', 0),
(37, 2, 'User.2.12.jpg', 0),
(38, 2, 'User.2.13.jpg', 0),
(39, 2, 'User.2.14.jpg', 0),
(40, 2, 'User.2.15.jpg', 0),
(41, 2, 'User.2.16.jpg', 0),
(59, 3, 'User.3.2.jpg', 0),
(60, 3, 'User.3.3.jpg', 0),
(61, 3, 'User.3.4.jpg', 0),
(62, 3, 'User.3.5.jpg', 0),
(63, 3, 'User.3.6.jpg', 0),
(64, 3, 'User.3.7.jpg', 0),
(65, 3, 'User.3.8.jpg', 0),
(66, 3, 'User.3.9.jpg', 0),
(67, 3, 'User.3.10.jpg', 0),
(68, 3, 'User.3.11.jpg', 0),
(69, 3, 'User.3.12.jpg', 0),
(70, 3, 'User.3.13.jpg', 0),
(71, 3, 'User.3.14.jpg', 0),
(72, 3, 'User.3.15.jpg', 0),
(73, 3, 'User.3.16.jpg', 0),
(74, 3, 'User.3.17.jpg', 0),
(75, 3, 'User.3.18.jpg', 0),
(76, 1, 'User.1.2.jpg', 0),
(77, 1, 'User.1.3.jpg', 0),
(78, 1, 'User.1.4.jpg', 0),
(79, 1, 'User.1.5.jpg', 0),
(80, 1, 'User.1.6.jpg', 0),
(81, 1, 'User.1.7.jpg', 0),
(82, 1, 'User.1.8.jpg', 0),
(83, 1, 'User.1.9.jpg', 0),
(84, 1, 'User.1.10.jpg', 0),
(85, 1, 'User.1.11.jpg', 0),
(86, 1, 'User.1.12.jpg', 0),
(87, 1, 'User.1.13.jpg', 0),
(88, 1, 'User.1.14.jpg', 0),
(89, 1, 'User.1.15.jpg', 0),
(90, 1, 'User.1.16.jpg', 0),
(91, 1, 'User.1.17.jpg', 0),
(92, 1, 'User.1.18.jpg', 0),
(93, 1, 'User.1.19.jpg', 0),
(94, 1, 'User.1.20.jpg', 0),
(95, 1, 'User.1.21.jpg', 0),
(96, 1, 'User.1.22.jpg', 0),
(97, 1, 'User.1.23.jpg', 0),
(98, 1, 'User.1.24.jpg', 0),
(99, 1, 'User.1.25.jpg', 0),
(100, 1, 'User.1.26.jpg', 0),
(101, 1, 'User.1.27.jpg', 0),
(102, 1, 'User.1.28.jpg', 0),
(103, 1, 'User.1.29.jpg', 0),
(104, 1, 'User.1.30.jpg', 0),
(105, 1, 'User.1.31.jpg', 0),
(106, 1, 'User.1.32.jpg', 0),
(107, 1, 'User.1.33.jpg', 0),
(108, 1, 'User.1.34.jpg', 0),
(109, 1, 'User.1.35.jpg', 0),
(110, 1, 'User.1.36.jpg', 0),
(111, 1, 'User.1.37.jpg', 0),
(112, 1, 'User.1.38.jpg', 0),
(113, 1, 'User.1.39.jpg', 0),
(114, 1, 'User.1.40.jpg', 0);
